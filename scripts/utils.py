import os
import numpy as np
import pandas as pd

project_path = os.getcwd()

PROJECT_PATH = project_path
DATA_PATH = PROJECT_PATH + "/raw_data/loan.csv"
PARQUET_PATH = PROJECT_PATH + "cohort.parquet"
OUTPUTS_PATH = PROJECT_PATH + "outputs/"
FIGURES_PATH = OUTPUTS_PATH +"figures/"
TABLES_PATH = OUTPUTS_PATH +"tables/"
MODELS_PATH = OUTPUTS_PATH + "models/"

HORIZONS_MONTHS = [12, 24, 36]

COMPLETED_STATUSES = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
    "Default",
]
EVENT_STATUSES = [
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
    "Default",
]

# remove columns that might lead to information leakage
LEAKY_PREFIXES = ("total_", "recover", "collection_", "last_pymnt", "out_prncp")
LEAKY_EXACT = {
    "loan_status",
    "pymnt_plan",
    "hardship_flag",
    "debt_settlement_flag",
    "last_credit_pull_d",
}

def ensure_dirs():
    os.makedirs(PROJECT_PATH + "outputs/figures", exist_ok=True)
    os.makedirs(PROJECT_PATH + "outputs/tables", exist_ok=True)
    os.makedirs(PROJECT_PATH + "outputs/models", exist_ok=True)

def parse_month_year(s):
    # LendingClub often uses "Dec-2011"
    return pd.to_datetime(s, format="%b-%Y", errors="coerce")

def pct_str_to_float(x):
    # "13.56%" -> 13.56
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x.endswith("%"):
        x = x[:-1]
    try:
        return float(x)
    except:
        return np.nan

def term_to_months(x):
    # "36 months" -> 36
    if pd.isna(x):
        return np.nan
    s = str(x)
    digits = "".join([c for c in s if c.isdigit()])
    return float(digits) if digits else np.nan

def drop_constant_and_duplicate_cols(X: pd.DataFrame, check_dups: bool = True, chunk_size: int = 512, seed: int = 0) -> pd.DataFrame:
    """
    Much faster than nunique + X.T.duplicated() on wide matrices.
    - constants: uses min/max (vectorized)
    - duplicates: uses 2 random projections (fingerprints) then verifies equality for collisions
    """
    # --- 1) drop constant columns ---
    mins = X.min(axis=0, numeric_only=True)
    maxs = X.max(axis=0, numeric_only=True)
    const_cols = mins.index[mins.eq(maxs)].tolist()
    if const_cols:
        X = X.drop(columns=const_cols, errors="ignore")

    if (not check_dups) or X.shape[1] <= 1:
        return X

    # --- 2) detect duplicate columns without transpose ---
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    r1 = rng.standard_normal(n).astype(np.float64)
    r2 = rng.standard_normal(n).astype(np.float64)

    cols = X.columns.to_list()
    fp1 = np.empty(len(cols), dtype=np.float64)
    fp2 = np.empty(len(cols), dtype=np.float64)

    for start in range(0, len(cols), chunk_size):
        block = X.iloc[:, start:start + chunk_size].to_numpy(dtype=np.float64, copy=False)
        fp1[start:start + block.shape[1]] = block.T @ r1
        fp2[start:start + block.shape[1]] = block.T @ r2

    seen = {}
    dup_cols = []
    for j, (a, b) in enumerate(zip(fp1, fp2)):
        key = (a, b)
        if key in seen:
            i0 = seen[key]
            if X.iloc[:, j].equals(X.iloc[:, i0]):
                dup_cols.append(cols[j])
        else:
            seen[key] = j

    if dup_cols:
        X = X.drop(columns=dup_cols, errors="ignore")

    return X


def pick_baseline_features(df: pd.DataFrame):
    # remove high-cardinality columns (emp_title, title, zip_code).
    candidates = [
        "loan_amnt",
        "term",
        "int_rate",
        "grade",       # keep grade OR sub_grade
        "emp_length",
        "home_ownership",
        "annual_inc",
        "verification_status",
        "purpose",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "earliest_cr_line",
        "application_type",
    ]
    return [c for c in candidates if c in df.columns]

def is_leaky_col(c: str) -> bool:
    if c in LEAKY_EXACT:
        return True
    return c.startswith(LEAKY_PREFIXES)
