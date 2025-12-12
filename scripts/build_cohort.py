import pandas as pd
from scripts.utils import (ensure_dirs,DATA_PATH,PARQUET_PATH,PROJECT_PATH, COMPLETED_STATUSES, EVENT_STATUSES, parse_month_year,pct_str_to_float,term_to_months,parse_month_year)

def main(DATA_PATH: str):
    ensure_dirs()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    df = df[df["loan_status"].isin(COMPLETED_STATUSES)].copy()
    df["event"] = df["loan_status"].isin(EVENT_STATUSES).astype(int)
    
    df["issue_d"] = parse_month_year(df["issue_d"])
    df["last_pymnt_d"] = parse_month_year(df["last_pymnt_d"])
    df = df.dropna(subset=["issue_d", "last_pymnt_d"]).copy()
    
    df["time_months"] = (df["last_pymnt_d"] - df["issue_d"]).dt.days / 30.44
    df = df[df["time_months"] > 0].copy()
    
    # basic transforms
    print("transform term dtype")
    if "term" in df.columns:
        df["term_months"] = df["term"].map(term_to_months)
        df = df.drop(columns=["term"])
        
    print("transform interest rate dtype")
    if "int_rate" in df.columns:
        df["int_rate_num"] = df["int_rate"].map(pct_str_to_float)
        df = df.drop(columns=["int_rate"])
        
    print("transform revol_util dtype")
    if "revol_util" in df.columns:
        df["revol_util_num"] = df["revol_util"].map(pct_str_to_float)
        df = df.drop(columns=["revol_util"])\
            
    print("transform earliet credit line dtype")
    if "earliest_cr_line" in df.columns:
        ecl = parse_month_year(df["earliest_cr_line"])
        df["credit_history_yrs"] = (df["issue_d"] - ecl).dt.days / 365.25
        df = df.drop(columns=["earliest_cr_line"])
        
    #dropping columns with more than 50% missing data
    pct_null = df.isnull().sum() / len(df)
    missing_features = pct_null[pct_null > 0.50].index
    print(missing_features)
    df = df.drop(missing_features, axis=1)
    
    #drop columns with no unique values
    cols_to_drop_constant = [col for col in df.columns if df[col].nunique() == 1]
    print("constant columns",cols_to_drop_constant)
    df = df.drop(cols_to_drop_constant, axis=1)
    
    print("number of columns in the dataset", len(df.columns))
    print('earliest issue date', df['issue_d'].min())
    print('latest issue date', df['issue_d'].max())
    
    df.to_parquet(PARQUET_PATH, index=False)
    print("saved:", PARQUET_PATH)
    print("shape:", df.shape)
    print("event rate:", round(df["event"].mean(), 4))

if __name__ == "__main__":
    main(DATA_PATH)
