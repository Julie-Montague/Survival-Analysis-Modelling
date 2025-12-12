import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

from scripts.utils import (
    PARQUET_PATH, FIGURES_PATH, TABLES_PATH, MODELS_PATH,
    HORIZONS_MONTHS, ensure_dirs, pick_baseline_features, is_leaky_col
)

#UTILITIES
def _safe_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d"]).copy()
    return df

def time_split_raw(df: pd.DataFrame, test_frac=0.2):
    cutoff = df["issue_d"].quantile(1 - test_frac)
    tr = df[df["issue_d"] <= cutoff].copy()
    te = df[df["issue_d"] > cutoff].copy()
    return tr, te, cutoff

def km_event_prob(df_y, horizon):
    """Observed P(event by horizon) via KM (handles censoring)."""
    kmf = KaplanMeierFitter()
    kmf.fit(df_y["time_months"], event_observed=df_y["event"])
    s = float(kmf.survival_function_at_times(horizon).values[0])
    return 1 - s

def calibration_table(df_y, pred, horizon, bins=10):
    tmp = df_y[["time_months", "event"]].copy()
    tmp["pred"] = pred
    tmp["bin"] = pd.qcut(tmp["pred"], q=bins, duplicates="drop")

    rows = []
    for b, g in tmp.groupby("bin", observed=True):
        rows.append({
            "bin": str(b),
            "mean_pred": float(g["pred"].mean()),
            "obs_event_prob": float(km_event_prob(g, horizon)),
            "n": int(len(g)),
        })

    out = pd.DataFrame(rows).sort_values("mean_pred").reset_index(drop=True)
    out["abs_calib_error"] = (out["mean_pred"] - out["obs_event_prob"]).abs()
    return out

def lift_table(df_y, pred, horizon, fracs=(0.05, 0.1, 0.2, 0.3)):
    tmp = df_y[["time_months", "event"]].copy()
    tmp["pred"] = pred
    tmp = tmp.sort_values("pred", ascending=False)

    base = km_event_prob(tmp, horizon)
    rows = []
    for f in fracs:
        k = max(1, int(len(tmp) * f))
        top = tmp.head(k)
        top_obs = km_event_prob(top, horizon)
        rows.append({
            "top_frac": f,
            "n_targeted": k,
            "obs_top": float(top_obs),
            "obs_all": float(base),
            "lift": float(top_obs / base) if base > 0 else np.nan
        })
    return pd.DataFrame(rows)

def plot_calib(calib, horizon, outpath):
    plt.figure()
    plt.plot(calib["mean_pred"], calib["obs_event_prob"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("Predicted P(default by horizon)")
    plt.ylabel("Observed P(default by horizon) (KM)")
    plt.title(f"Calibration at {horizon} months")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_lift(lift_df, horizon, outpath):
    plt.figure()
    plt.plot(lift_df["top_frac"], lift_df["lift"], marker="o")
    plt.xlabel("Top fraction targeted")
    plt.ylabel("Lift (KM top / KM overall)")
    plt.title(f"Lift at {horizon} months")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def km_curve_long(df_y, group_series, curve_name, time_grid):
    """Long-form KM curves"""
    tmp = df_y.copy()
    tmp["group"] = group_series.astype("string")

    rows = []
    kmf = KaplanMeierFitter()
    for g, gdf in tmp.groupby("group", observed=True):
        kmf.fit(gdf["time_months"], event_observed=gdf["event"])
        s = kmf.survival_function_at_times(time_grid).values
        part = pd.DataFrame({
            "curve": curve_name,
            "group": str(g),
            "time_months": time_grid,
            "survival_prob": s,
        })
        part["cum_default_prob"] = 1.0 - part["survival_prob"]
        rows.append(part)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def pick_meta_cols(df):
    candidates = ["loan_id", "id", "member_id", "issue_d", "grade", "sub_grade", "term", "purpose", "int_rate", "loan_amnt"]
    return [c for c in candidates if c in df.columns]

# -------------------------
# Preprocessing
# -------------------------

def build_Xy_with_preproc(df_raw: pd.DataFrame, preproc: dict):
    """
    Rebuild covariate matrix using training artifact:
      - baseline feature selection
      - leaky-col removal
      - numeric median imputation from train
      - top-K category mapping from train
      - one-hot + align to feature_columns
    """
    y = df_raw[["time_months", "event"]].copy()

    feats = pick_baseline_features(df_raw)
    X = df_raw[feats].copy()
    X = X[[c for c in X.columns if not is_leaky_col(c)]].copy()

    num_cols = preproc.get("num_cols", [])
    cat_cols = preproc.get("cat_cols", [])

    # numeric medians from train
    med = preproc.get("num_medians", {})
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X[c] = X[c].fillna(med.get(c, np.nan))

    # categoricals: apply train top-level mapping AND train category order
    top_levels = preproc.get("top_levels", {})
    for c in cat_cols:
        if c in X.columns:
            s = X[c].astype("string").fillna("Missing")
            top_list = top_levels.get(c, [])
            top_set = set(top_list)

            # unseen -> Other, Missing stays Missing
            s = s.where((s == "Missing") | (s.isin(top_set)), "Other")

            # match training: "Other" baseline first, then top levels (excluding Other), then Missing
            cats = ["Other"] + [v for v in top_list if v != "Other"]
            if "Missing" not in cats:
                cats.append("Missing")
            seen = set()
            cats = [x for x in cats if not (x in seen or seen.add(x))]

            X[c] = pd.Categorical(s, categories=cats)

    # one-hot (drop_first MUST drop "Other" due to category order above)
    if cat_cols:
        cols = [c for c in cat_cols if c in X.columns]
        try:
            X = pd.get_dummies(X, columns=cols, drop_first=True, dtype=np.uint8)
        except TypeError:
            X = pd.get_dummies(X, columns=cols, drop_first=True).astype(np.uint8)

    # align to train
    feature_cols = preproc["feature_columns"]
    X = X.reindex(columns=feature_cols, fill_value=0)

    return X, y


# Predictions
def predict_pd_by_horizon_cox(cph, X, horizons):
    sf = cph.predict_survival_function(X, times=horizons)  # rows=times, cols=individuals
    return {h: (1.0 - sf.loc[h].values) for h in horizons}

def load_rsf(rsfpkl):
    if not rsfpkl or (not os.path.exists(rsfpkl)):
        return None
    try:
        with open(rsfpkl, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def predict_pd_by_horizon_rsf(rsf, X_np, horizons):
    surv_fns = rsf.predict_survival_function(X_np, return_array=False)
    out = {}
    for h in horizons:
        out[h] = 1.0 - np.array([fn(h) for fn in surv_fns], dtype=float)
    return out

#Main code
def main(cohort_path, cox_model_path, preproc_path, rsf_model_path, test_frac, bins):
    ensure_dirs()

    df = pd.read_parquet(cohort_path)
    df = _safe_dates(df)

    df_tr_raw, df_te_raw, cutoff = time_split_raw(df, test_frac=test_frac)
    print("split cutoff(issue_d):", cutoff.date())
    print("train_raw:", df_tr_raw.shape, "test_raw:", df_te_raw.shape)

    with open(cox_model_path, "rb") as f:
        cph = pickle.load(f)
    with open(preproc_path, "rb") as f:
        preproc = pickle.load(f)

    X_train, y_train = build_Xy_with_preproc(df_tr_raw, preproc)
    X_test, y_test = build_Xy_with_preproc(df_te_raw, preproc)

    horizons = [float(h) for h in HORIZONS_MONTHS]

    # ---- COX ----
    cox_pd = predict_pd_by_horizon_cox(cph, X_test, horizons)
    cox_risk = cph.predict_partial_hazard(X_test).values.ravel()

    # bins based on PD@first horizon (e.g., 12m)
    cox_bin = pd.qcut(pd.Series(cox_pd[horizons[0]]), q=5, duplicates="drop", labels=False)
    cox_bin = cox_bin.astype("Int64").map(lambda x: f"Q{int(x)+1}" if pd.notna(x) else "NA")

    # ---- RSF ----
    rsf = load_rsf(rsf_model_path)
    have_rsf = rsf is not None
    if have_rsf:
        X_test_np = X_test.astype("float32").values
        rsf_pd = predict_pd_by_horizon_rsf(rsf, X_test_np, horizons)
        rsf_risk = rsf.predict(X_test_np)

        rsf_bin = pd.qcut(pd.Series(rsf_pd[horizons[0]]), q=5, duplicates="drop", labels=False)
        rsf_bin = rsf_bin.astype("Int64").map(lambda x: f"Q{int(x)+1}" if pd.notna(x) else "NA")
        print("RSF loaded:", rsf_model_path)
    else:
        rsf_pd, rsf_risk, rsf_bin = None, None, None
        print("RSF model not found/loaded. Skipping RSF exports. Expected at:", rsf_model_path)

    # ---- Table 1: cohort overview ----
    df_over = df[["issue_d", "time_months", "event"]].copy()
    df_over["issue_year"] = df_over["issue_d"].dt.year.astype("Int64")
    cohort_overview = (
        df_over.groupby("issue_year", dropna=True)
        .agg(n_loans=("event", "size"),
             event_rate=("event", "mean"),
             median_time_months=("time_months", "median"))
        .reset_index()
    )
    cohort_overview.to_csv(TABLES_PATH + "cohort_overview.csv", index=False)

    # ---- Table 2: KM curves (overall + by risk bins) ----
    max_t = int(min(np.ceil(y_test["time_months"].max()), 60))
    time_grid = np.arange(0, max_t + 1, 1, dtype=float)

    km_all = km_curve_long(y_test, pd.Series(["Overall"] * len(y_test)), "overall", time_grid)
    km_cox = km_curve_long(y_test, cox_bin.fillna("NA"), "cox_risk_bin", time_grid)
    km_list = [km_all, km_cox]

    if have_rsf:
        km_rsf = km_curve_long(y_test, rsf_bin.fillna("NA"), "rsf_risk_bin", time_grid)
        km_list.append(km_rsf)

    km_long = pd.concat([x for x in km_list if len(x) > 0], ignore_index=True)
    km_long.to_csv(TABLES_PATH + "km_curves_long.csv", index=False)

    # ---- Table 3: loan-level predictions (test only) ----
    meta_cols = pick_meta_cols(df_te_raw)
    meta = df_te_raw[meta_cols].copy() if meta_cols else pd.DataFrame({"row_id": df_te_raw.index})
    if "issue_d" in meta.columns:
        meta["issue_d"] = pd.to_datetime(meta["issue_d"], errors="coerce")

    pred_out = pd.concat([meta.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    pred_out["cox_risk"] = cox_risk
    pred_out["cox_risk_bin"] = cox_bin.values
    for h in horizons:
        pred_out[f"cox_pd_{int(h)}m"] = cox_pd[h]

    if have_rsf:
        pred_out["rsf_risk"] = rsf_risk
        pred_out["rsf_risk_bin"] = rsf_bin.values
        for h in horizons:
            pred_out[f"rsf_pd_{int(h)}m"] = rsf_pd[h]

    pred_out.to_csv(TABLES_PATH + "predictions_test_loan_level.csv", index=False)

    # ---- Calibration + lift (Cox + RSF) ----
    def run_calib_lift(model_name, pd_dict):
        kpi_rows = []
        for h in horizons:
            pred = pd_dict[h]

            obs_all = km_event_prob(y_test, h)
            pred_mean = float(np.mean(pred))
            pred_sum = float(np.sum(pred))

            tmp = y_test.copy()
            tmp["pred"] = pred
            tmp = tmp.sort_values("pred", ascending=False)
            top10 = tmp.head(max(1, int(0.10 * len(tmp))))
            obs_top10 = km_event_prob(top10, h)
            lift10 = obs_top10 / obs_all if obs_all > 0 else np.nan

            kpi_rows.append({
                "model": model_name,
                "horizon_months": int(h),
                "n_test": int(len(tmp)),
                "pred_mean_pd": pred_mean,
                "pred_expected_defaults": pred_sum,
                "obs_km_pd": float(obs_all),
                "obs_top10_km_pd": float(obs_top10),
                "top10_lift": float(lift10),
            })

            calib = calibration_table(y_test, pred, horizon=h, bins=bins)
            calib["model"] = model_name
            calib["horizon_months"] = int(h)
            calib.to_csv(TABLES_PATH + f"calibration_{model_name}_{int(h)}m.csv", index=False)
            plot_calib(calib, h, FIGURES_PATH + f"calibration_{model_name}_{int(h)}m.png")

            lift = lift_table(y_test, pred, horizon=h)
            lift["model"] = model_name
            lift["horizon_months"] = int(h)
            lift.to_csv(TABLES_PATH + f"lift_{model_name}_{int(h)}m.csv", index=False)
            plot_lift(lift, h, FIGURES_PATH + f"lift_{model_name}_{int(h)}m.png")

        return pd.DataFrame(kpi_rows)

    kpi_all = [run_calib_lift("cox", cox_pd)]
    if have_rsf:
        kpi_all.append(run_calib_lift("rsf", rsf_pd))

    horizon_kpis = pd.concat(kpi_all, ignore_index=True)
    horizon_kpis.to_csv(TABLES_PATH + "horizon_kpis.csv", index=False)

    # ---- Metrics summary ----
    rows = []

    c_cox = concordance_index(y_test["time_months"], -cox_risk, y_test["event"])
    rows.append({"model": "cox", "metric": "c_index", "horizon_months": "", "value": float(c_cox)})

    if have_rsf:
        c_rsf = concordance_index(y_test["time_months"], -rsf_risk, y_test["event"])
        rows.append({"model": "rsf", "metric": "c_index", "horizon_months": "", "value": float(c_rsf)})

    # Time-dependent AUC + IBS
    try:

        ytr_s = Surv.from_arrays(event=y_train["event"].astype(bool).values, time=y_train["time_months"].values)
        yte_s = Surv.from_arrays(event=y_test["event"].astype(bool).values, time=y_test["time_months"].values)

        tmax_test = float(y_test["time_months"].max())
        times_all = np.array([12.0, 24.0, 36.0])

        # keep only horizons that are within test follow-up
        times = times_all[times_all < tmax_test]

        print(f"AUC eval times used (<= test follow-up {tmax_test:.2f}):", times)

        if len(times) > 0:
            auc, _ = cumulative_dynamic_auc(ytr_s, yte_s, cox_risk, times)
            for t, a in zip(times, auc):
                rows.append({"model": "cox", "metric": "auc_td", "horizon_months": int(t), "value": float(a)})
        else:
            print("Skipping time-dependent AUC: not enough test follow-up.")


        print("IBS SCORE")

        tmin = float(y_test["time_months"].min())
        tmax = float(y_test["time_months"].max())

        # upper bound is effectively exclusive
        t_end = float(np.nextafter(tmax, -np.inf))

        grid_start = max(1.0, float(np.ceil(tmin)))
        grid_end = float(np.floor(t_end))

        if grid_end <= grid_start:
            raise ValueError(f"Not enough follow-up for IBS grid: tmin={tmin:.3f}, tmax={tmax:.3f}")

        grid = np.arange(grid_start, grid_end + 1.0, 1.0, dtype=float)
        print(f"IBS grid: {grid[0]}..{grid[-1]} (test follow-up max={tmax:.2f})")

        cox_sf = cph.predict_survival_function(X_test, times=grid).T.values
        ibs = integrated_brier_score(ytr_s, yte_s, cox_sf, grid)
        rows.append({"model": "cox", "metric": "ibs", "horizon_months": "", "value": float(ibs)})

        if have_rsf:
            X_test_np = X_test.astype("float32").values
            auc2, _ = cumulative_dynamic_auc(ytr_s, yte_s, rsf_risk, times)
            for t, a in zip(times, auc2):
                rows.append({"model": "rsf", "metric": "auc_td", "horizon_months": int(t), "value": float(a)})

            surv_fns = rsf.predict_survival_function(X_test_np, return_array=False)
            rsf_sf = np.row_stack([fn(grid) for fn in surv_fns])
            ibs2 = integrated_brier_score(ytr_s, yte_s, rsf_sf, grid)
            rows.append({"model": "rsf", "metric": "ibs", "horizon_months": "", "value": float(ibs2)})

    except Exception as e:
        print("scikit-survival not available or failed; metrics_summary will include only c-index.")
        print("Details:", repr(e))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES_PATH + "metrics_summary.csv", index=False)

    print("DONE. Tables saved to:", TABLES_PATH, "Figures saved to:", FIGURES_PATH)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default=PARQUET_PATH)
    p.add_argument("--cox_model", default=MODELS_PATH + "cox_model.pkl")
    p.add_argument("--preproc", default=MODELS_PATH + "cox_preprocess.pkl")
    p.add_argument("--rsf_model", default=MODELS_PATH + "rsf_model.pkl")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--bins", type=int, default=10)
    args = p.parse_args()

    main(args.cohort, args.cox_model, args.preproc, args.rsf_model, args.test_frac, args.bins)
