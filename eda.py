import pandas as pd
import numpy as np


def _detect_outliers_iqr(series: pd.Series) -> int:
    """Return count of IQR-based outliers for a numeric series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def run_eda(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        raise ValueError("EDA failed: DataFrame is empty or None")

    eda = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "column_types": {},
        "missing_values": {},
        "numeric_summary": {},
        "outliers": {},
        "correlations": None
    }

    for col in df.columns:
        eda["column_types"][col] = str(df[col].dtype)
        eda["missing_values"][col] = int(df[col].isnull().sum())

    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:
        desc = numeric_df.describe().T
        for col in desc.index:
            eda["numeric_summary"][col] = {
                "mean": float(desc.loc[col, "mean"]),
                "std": float(desc.loc[col, "std"]),
                "min": float(desc.loc[col, "min"]),
                "max": float(desc.loc[col, "max"]),
            }
            eda["outliers"][col] = _detect_outliers_iqr(numeric_df[col])

        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            eda["correlations"] = corr.round(3).to_dict()

    return eda
