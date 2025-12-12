import argparse
import pickle
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from scripts.utils import (PARQUET_PATH, MODELS_PATH, ensure_dirs)
from scripts.eval_business import _safe_dates, time_split_raw, build_Xy_with_preproc

def main(cohort_path, preproc_path, out_rsf):
    ensure_dirs()

    df = pd.read_parquet(cohort_path)
    df = _safe_dates(df)

    # train-test split
    df_tr_raw, df_te_raw, cutoff = time_split_raw(df, test_frac=0.2)
    print("split cutoff(issue_d):", cutoff.date())
    print("train_raw:", df_tr_raw.shape, "test_raw:", df_te_raw.shape)

    with open(preproc_path, "rb") as f:
        preproc = pickle.load(f)

    X_train, y_train_df = build_Xy_with_preproc(df_tr_raw, preproc)

    y_train = Surv.from_arrays(event=y_train_df["event"].astype(bool).values,
                               time=y_train_df["time_months"].values)

    Xtr = X_train.astype("float32").values

    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    print("training rsf model")
    rsf.fit(Xtr, y_train)

    with open(out_rsf, "wb") as f:
        pickle.dump(rsf, f)

    print("saved rsf model:", out_rsf)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default=PARQUET_PATH)
    p.add_argument("--preproc", default=MODELS_PATH + "cox_preprocess.pkl")
    p.add_argument("--out_rsf", default=MODELS_PATH + "rsf_model.pkl")
    args = p.parse_args()
    main(args.cohort, args.preproc, args.out_rsf)
