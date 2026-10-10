"""
Data quality pipeline for the Online Retail II churn project.
Each function does ONE job — easier to test and debug.
"""

from pathlib import Path

import pandas as pd

NON_PRODUCT_CODES = [
    "POST", "DOT", "M", "BANK CHARGES",
    "AMAZONFEE", "CRUK", "PADS", "C2", "ADJUST",
]


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert InvoiceDate from text to datetime."""
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def flag_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where Invoice starts with 'C' (cancellation)."""
    df = df.copy()
    df["is_cancellation"] = df["Invoice"].astype(str).str.startswith("C")
    return df


def flag_non_products(df: pd.DataFrame) -> pd.DataFrame:
    """Flag StockCodes that are not real products (postage, fees, etc.)."""
    df = df.copy()
    df["is_non_product"] = df["StockCode"].isin(NON_PRODUCT_CODES)
    return df


def remove_bad_debt_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop accounting adjustment rows (StockCode == 'B')."""
    return df[df["StockCode"] != "B"].copy()


def remove_cross_sheet_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows across sheets.
    IMPORTANT: 'source_sheet' is excluded from the subset,
    otherwise true duplicates get missed (see Phase 3 finding).
    """
    dedup_cols = [
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
    ]
    return df.drop_duplicates(subset=dedup_cols, keep="first")


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run the full cleaning pipeline and return (cleaned_df, quality_report).
    """
    report = {"rows_before": len(df)}

    df = convert_dtypes(df)
    df = flag_cancellations(df)
    df = flag_non_products(df)

    report["bad_debt_rows_removed"] = (df["StockCode"] == "B").sum()
    df = remove_bad_debt_rows(df)

    rows_before_dedup = len(df)
    df = remove_cross_sheet_duplicates(df)
    report["duplicate_rows_removed"] = rows_before_dedup - len(df)

    report["rows_after"] = len(df)
    report["missing_customer_id"] = df["Customer ID"].isnull().sum()
    report["missing_customer_id_pct"] = round(
        df["Customer ID"].isnull().mean() * 100, 2
    )
    report["cancellation_rows"] = df["is_cancellation"].sum()
    report["non_product_rows"] = df["is_non_product"].sum()

    return df, report


def print_report(report: dict) -> None:
    """Pretty-print the quality report."""
    print("=" * 50)
    print("DATA QUALITY REPORT")
    print("=" * 50)
    for key, value in report.items():
        print(f"{key:30s}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    RAW_PATH = Path("D:/SUBJECTS/Data_Analyst/AnalystProjects/customer_churn_prediction/data/raw/online_retail_II.csv")
    PROCESSED_PATH = Path("D:/SUBJECTS/Data_Analyst/AnalystProjects/customer_churn_prediction/data/processed/online_retail_II_clean.csv")

    raw_df = pd.read_csv(RAW_PATH)
    clean_df, quality_report = clean_data(raw_df)

    print_report(quality_report)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved cleaned data to {PROCESSED_PATH}")