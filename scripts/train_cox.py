import argparse
import pickle
import numpy as np
import pandas as pd
import time
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scripts.utils import (PARQUET_PATH,TABLES_PATH,MODELS_PATH,ensure_dirs, pick_baseline_features, drop_constant_and_duplicate_cols, is_leaky_col)

def time_split_raw(df: pd.DataFrame, test_frac=0.2):
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d"])
    cutoff = df["issue_d"].quantile(1 - test_frac)
    tr = df[df["issue_d"] <= cutoff].copy()
    te = df[df["issue_d"] > cutoff].copy()
    return tr, te, cutoff

def cap_top_k(s: pd.Series, k: int = 5) -> pd.Series:
    s = s.astype("string").fillna("Missing")
    top = s.value_counts(dropna=False).head(k).index
    return s.where(s.isin(top), "Other")

def _cap_top_k_train_test(
    s_train: pd.Series,
    s_test: pd.Series,
    k: int = 5
):
    # missing stays as its own bucket; everything else not in top-k => Other
    tr = s_train.astype("string").fillna("Missing")
    te = s_test.astype("string").fillna("Missing")

    top = tr.value_counts(dropna=False).head(k).index.tolist()
    top_set = set(top)

    def map_series(s):
        return s.where((s == "Missing") | (s.isin(top_set)), "Other")

    tr_m = map_series(tr)
    te_m = map_series(te)

    # Control baseline category for drop_first=True:
    # put "Other" first so coefficients interpret as vs Other.
    cats = ["Other"] + [c for c in top if c not in ("Other",)]  # keep top order
    if "Missing" not in cats:
        cats.append("Missing")
    # unique while preserving order
    seen = set()
    cats = [x for x in cats if not (x in seen or seen.add(x))]

    tr_m = pd.Categorical(tr_m, categories=cats)
    te_m = pd.Categorical(te_m, categories=cats)

    return tr_m, te_m, top

def prep_train_test_matrix(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    top_k: int = 5
):
    t0 = time.perf_counter()

    # targets
    y_tr = df_train_raw[["time_months", "event"]].copy()
    y_te = df_test_raw[["time_months", "event"]].copy()

    feats = pick_baseline_features(df_train_raw)
    X_tr = df_train_raw[feats].copy()
    X_te = df_test_raw[feats].copy()

    # leakage guard
    keep = [c for c in X_tr.columns if not is_leaky_col(c)]
    X_tr = X_tr[keep].copy()
    X_te = X_te[keep].copy()

    # split num/cat based on TRAIN types (important)
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_tr.columns if c not in num_cols]

    print("numerical columns :", num_cols)
    print("categorical columns:", cat_cols)

    # numeric impute (train median, apply to test)
    if num_cols:
        X_tr[num_cols] = X_tr[num_cols].replace([np.inf, -np.inf], np.nan)
        X_te[num_cols] = X_te[num_cols].replace([np.inf, -np.inf], np.nan)
        med = X_tr[num_cols].median()
        X_tr[num_cols] = X_tr[num_cols].fillna(med)
        X_te[num_cols] = X_te[num_cols].fillna(med)

    # cap categoricals to top-k (train-only) + apply same mapping to test
    top_levels = {}
    if cat_cols:
        card = X_tr[cat_cols].nunique(dropna=False).sort_values(ascending=False)
        print("Top categorical cardinalities (train):\n", card.head(15))

        for c in cat_cols:
            tr_c, te_c, top = _cap_top_k_train_test(X_tr[c], X_te[c], k=top_k)
            X_tr[c] = tr_c
            X_te[c] = te_c
            top_levels[c] = top

        # one-hot encoding
        print("one hot encoding")
        try:
            X_tr = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True, dtype=np.uint8)
            X_te = pd.get_dummies(X_te, columns=cat_cols, drop_first=True, dtype=np.uint8)
        except TypeError:
            X_tr = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True).astype(np.uint8)
            X_te = pd.get_dummies(X_te, columns=cat_cols, drop_first=True).astype(np.uint8)

    # drop constant/duplicate columns based on TRAIN ONLY; align TEST
    print('drop redundant columns')
    X_tr = drop_constant_and_duplicate_cols(X_tr)
    keep_cols = X_tr.columns.tolist()
    print("length of columns",len(keep_cols))
    X_te = X_te.reindex(columns=keep_cols, fill_value=0)

    print('concat x and y')
    train = pd.concat([y_tr, X_tr], axis=1)
    test  = pd.concat([y_te, X_te], axis=1)

    preproc = {
        "top_k": top_k,
        "top_levels": top_levels,
        "num_medians": med.to_dict() if num_cols else {},
        "feature_columns": keep_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    print(f"matrix prep completed: train={train.shape}, test={test.shape} in {time.perf_counter()-t0:.2f}s")
    return train, test, preproc

def main(cohort_path: str, penalizer: float, test_frac: float, out_model: str, out_preproc: str, check_ph: bool):
    ensure_dirs()
    df = pd.read_parquet(cohort_path)

    print("split the data (raw, before one-hot)")
    df_tr_raw, df_te_raw, cutoff = time_split_raw(df, test_frac=test_frac)
    print("split cutoff(issue_d):", cutoff.date())
    print("train_raw:", df_tr_raw.shape, "test_raw:", df_te_raw.shape)

    print("matrix prepping (top-5 cap, train-only mapping)")
    train, test, preproc = prep_train_test_matrix(df_tr_raw, df_te_raw, top_k=5)

    cph = CoxPHFitter(penalizer=penalizer)
    print("fitting cox model")
    cph.fit(train, duration_col="time_months", event_col="event")
    
    hr = cph.summary.sort_values("p").reset_index().rename(columns={"covariate": "feature"})
    print(hr)
    out = pd.DataFrame({
        "feature": hr["feature"],
        "hazard_ratio": hr["exp(coef)"],
        "ci_low": hr["exp(coef) lower 95%"],
        "ci_high": hr["exp(coef) upper 95%"],
        "p_value": hr["p"],
    })
    out["direction"] = np.where(out["hazard_ratio"] >= 1, "↑ risk", "↓ risk")
    out["abs_log_hr"] = np.abs(np.log(out["hazard_ratio"]))
    out = out.sort_values("abs_log_hr", ascending=False)

    out.to_csv(TABLES_PATH + "cox_hazard_ratios.csv", index=False)
    print("saved cox_hazard_ratios.csv")

    # c-index (risk = partial hazard)
    Xtr_cov = train.drop(columns=["time_months", "event"])
    Xte_cov = test.drop(columns=["time_months", "event"])

    train_risk = cph.predict_partial_hazard(Xtr_cov).values.ravel()
    test_risk  = cph.predict_partial_hazard(Xte_cov).values.ravel()

    c_train = concordance_index(train["time_months"], -train_risk, train["event"])
    c_test  = concordance_index(test["time_months"],  -test_risk,  test["event"])
    print("c-index train:", round(c_train, 4))
    print("c-index test :", round(c_test, 4))

    # save model
    with open(out_model, "wb") as f:
        pickle.dump(cph, f)
    print("saved model:", out_model)

    # save preproc artifact
    with open(out_preproc, "wb") as f:
        pickle.dump(preproc, f)
    print("saved preproc:", out_preproc)

    if check_ph:
        print("running PH checks...")
        cph.check_assumptions(train, p_value_threshold=0.05, show_plots=True)
        
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default=PARQUET_PATH)
    p.add_argument("--penalizer", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--out_model", default= MODELS_PATH + "cox_model.pkl")
    p.add_argument("--out_preproc", default= MODELS_PATH + "cox_preprocess.pkl")
    p.add_argument("--check_ph", action="store_true")
    args = p.parse_args()

    main(args.cohort, args.penalizer, args.test_frac, args.out_model, args.out_preproc, args.check_ph)