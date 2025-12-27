import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pandas.api.types import is_datetime64_any_dtype


class ForecastError(Exception):
    pass


def _validate_timeseries(df, date_col, target_col):
    if date_col not in df.columns:
        raise ForecastError(f"Date column '{date_col}' not found")

    if target_col not in df.columns:
        raise ForecastError(f"Target column '{target_col}' not found")

    if not is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            raise ForecastError("Date column cannot be parsed as datetime")

    if not np.issubdtype(df[target_col].dtype, np.number):
        raise ForecastError("Target column must be numeric")

    clean = df[[date_col, target_col]].dropna()
    if len(clean) < 10:
        raise ForecastError("Not enough data points for forecasting")

    return clean.sort_values(date_col)


def run_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    steps: int = 10,
    order: tuple = (1, 1, 1)
) -> dict:
    """
    Returns forecast values + confidence intervals.
    """
    ts = _validate_timeseries(df, date_col, target_col)
    ts.set_index(date_col, inplace=True)

    try:
        model = ARIMA(ts[target_col], order=order)
        fitted = model.fit()
    except Exception as e:
        raise ForecastError(f"ARIMA training failed: {str(e)}")

    forecast_res = fitted.get_forecast(steps=steps)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    return {
        "model": "ARIMA",
        "order": order,
        "steps": steps,
        "forecast": forecast.round(3).to_dict(),
        "confidence_interval": {
            "lower": conf_int.iloc[:, 0].round(3).to_dict(),
            "upper": conf_int.iloc[:, 1].round(3).to_dict(),
        }
    }
