import pandas as pd
import matplotlib.pyplot as plt
from scripts.utils import PROJECT_PATH,ensure_dirs,parse_month_year,pct_str_to_float


def main(csv_path: str):
    ensure_dirs()
    DATA_PATH = PROJECT_PATH + csv_path
    FIGURES_PATH = PROJECT_PATH+"outputs/figures/"
    TABLES_PATH = PROJECT_PATH+"outputs/tables/"
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # quick shape + missingness
    print("shape:", df.shape)
    miss = df.isna().mean().sort_values(ascending=False)

    top_miss = miss.head(30)
    plt.figure()
    plt.bar(top_miss.index.astype(str), top_miss.values)
    plt.xticks(rotation=90)
    plt.ylabel("missing fraction")
    plt.title("Missingness (top 30 columns)")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH + "raw_missingness_top30.png", dpi=200)
    plt.close()

    #loan_status distribution
    if "loan_status" in df.columns:
        props = df["loan_status"].value_counts(normalize=True)
        props.to_csv(TABLES_PATH + "raw_loan_status_proportions.csv")

        plt.figure()
        props.head(12).sort_values().plot(kind="barh")
        plt.title("loan_status proportions (top 12)")
        plt.xlabel("proportion")
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + "raw_loan_status_barh.png", dpi=200)
        plt.close()

    # 3) issue date coverage
    if "issue_d" in df.columns:
        issue = parse_month_year(df["issue_d"])
        issue_year = issue.dt.year
        plt.figure()
        issue_year.value_counts().sort_index().plot(kind="bar")
        plt.title("Loans by issue year")
        plt.xlabel("year")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + "raw_issue_year_counts.png", dpi=200)
        plt.close()

    # 4) a few numeric distributions
    def hist(col, transform=None, fname=None, title=None):
        if col not in df.columns:
            return
        x = df[col]
        if transform is not None:
            x = x.map(transform)
        x = pd.to_numeric(x, errors="coerce").dropna()
        if len(x) == 0:
            return
        plt.figure()
        plt.hist(x, bins=60)
        plt.title(title or f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(fname or FIGURES_PATH + f"raw_hist_{col}.png", dpi=200)
        plt.close()

    hist("loan_amnt", fname=FIGURES_PATH + "raw_hist_loan_amnt.png")
    hist("int_rate", transform=pct_str_to_float, fname=FIGURES_PATH + "raw_hist_int_rate.png", title="Distribution: int_rate (%)")
    hist("annual_inc", fname=FIGURES_PATH +"raw_hist_annual_inc.png")
    hist("dti", fname=FIGURES_PATH + "raw_hist_dti.png")

    # 5) top categories (lightly)
    for col in ["grade", "purpose", "home_ownership", "verification_status", "addr_state"]:
        if col not in df.columns:
            continue
        vc = df[col].astype(str).value_counts().head(12).sort_values()
        plt.figure()
        vc.plot(kind="barh")
        plt.title(f"Top {len(vc)}: {col}")
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + f"raw_top_{col}.png", dpi=200)
        plt.close()

    print("saved figures in outputs/figures and tables in outputs/tables")

if __name__ == "__main__":
    main("loan.csv")
