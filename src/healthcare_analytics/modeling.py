"""Lightweight forecasting and anomaly detection using aggregate tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _seasonal_trend_prediction(y: np.ndarray, months: pd.Series, target_index: int) -> float:
    x = np.arange(len(y), dtype=float)
    if len(y) == 1:
        trend_forecast = float(y[0])
        seasonal_anchor = float(y[0])
    else:
        slope, intercept = np.polyfit(x, y, 1)
        trend_forecast = float(intercept + slope * target_index)
        target_month = months.iloc[target_index].month
        same_month_values = y[months.iloc[: len(y)].dt.month.to_numpy() == target_month]
        seasonal_anchor = float(same_month_values.mean()) if len(same_month_values) else float(y.mean())
    return max(0.0, 0.72 * trend_forecast + 0.28 * seasonal_anchor)


def forecast_monthly_cases(executive_overview: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Forecast total monthly trauma case volume with a transparent trend model."""

    if executive_overview.empty:
        return pd.DataFrame(columns=["month", "forecast_total_cases", "lower_bound", "upper_bound", "model"])

    history = executive_overview.copy()
    history["month_dt"] = pd.to_datetime(history["month"])
    history = history.sort_values("month_dt").reset_index(drop=True)
    y = history["total_cases"].astype(float).to_numpy()
    x = np.arange(len(history), dtype=float)

    if len(history) == 1:
        slope = 0.0
        intercept = y[0]
        residual_std = max(1.0, y[0] * 0.08)
    else:
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        residual_std = float(np.std(y - fitted)) or max(1.0, float(np.mean(y)) * 0.05)

    last_month = history["month_dt"].max()
    rows = []
    for step in range(1, horizon + 1):
        next_month = last_month + pd.DateOffset(months=step)
        trend_forecast = intercept + slope * (len(history) - 1 + step)
        same_month_history = history.loc[history["month_dt"].dt.month == next_month.month, "total_cases"]
        seasonal_anchor = float(same_month_history.mean()) if not same_month_history.empty else float(np.mean(y))
        forecast = max(0.0, 0.72 * trend_forecast + 0.28 * seasonal_anchor)
        rows.append(
            {
                "month": next_month.strftime("%Y-%m"),
                "forecast_total_cases": int(round(forecast)),
                "lower_bound": int(round(max(0.0, forecast - 1.64 * residual_std))),
                "upper_bound": int(round(forecast + 1.64 * residual_std)),
                "model": "linear_trend_with_seasonal_anchor",
            }
        )
    return pd.DataFrame(rows)


def compare_forecasting_models(executive_overview: pd.DataFrame, min_train_periods: int = 6) -> pd.DataFrame:
    """Backtest simple forecasting options and keep the transparent model as default.

    The comparison is intentionally lightweight so the project remains easy to
    run locally. Random forest is included only when scikit-learn is installed.
    """

    columns = [
        "model",
        "test_periods",
        "mae",
        "rmse",
        "mape",
        "selected_default",
        "notes",
    ]
    if executive_overview.empty or len(executive_overview) < min_train_periods + 2:
        return pd.DataFrame(columns=columns)

    history = executive_overview.copy()
    history["month_dt"] = pd.to_datetime(history["month"])
    history = history.sort_values("month_dt").reset_index(drop=True)
    y_all = history["total_cases"].astype(float).to_numpy()
    months = history["month_dt"]
    start = min_train_periods

    predictions: dict[str, list[float]] = {
        "previous_month_baseline": [],
        "three_month_moving_average": [],
        "linear_trend_with_seasonal_anchor": [],
    }
    actuals: list[float] = []

    random_forest_predictions: list[float] = []
    random_forest_available = True
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        RandomForestRegressor = None  # type: ignore[assignment]
        random_forest_available = False

    for target_index in range(start, len(history)):
        train_y = y_all[:target_index]
        actuals.append(float(y_all[target_index]))
        predictions["previous_month_baseline"].append(float(train_y[-1]))
        predictions["three_month_moving_average"].append(float(train_y[-3:].mean()))
        predictions["linear_trend_with_seasonal_anchor"].append(
            _seasonal_trend_prediction(train_y, months, target_index)
        )

        if RandomForestRegressor is not None:
            train_frame = pd.DataFrame(
                {
                    "t": np.arange(target_index, dtype=float),
                    "month": months.iloc[:target_index].dt.month.astype(float),
                }
            )
            target_frame = pd.DataFrame(
                {
                    "t": [float(target_index)],
                    "month": [float(months.iloc[target_index].month)],
                }
            )
            model = RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=42)
            model.fit(train_frame, train_y)
            random_forest_predictions.append(float(model.predict(target_frame)[0]))

    if random_forest_available:
        predictions["random_forest_regressor"] = random_forest_predictions

    actual_array = np.asarray(actuals, dtype=float)
    rows = []
    for model_name, model_predictions in predictions.items():
        predicted_array = np.asarray(model_predictions, dtype=float)
        errors = actual_array - predicted_array
        rows.append(
            {
                "model": model_name,
                "test_periods": len(actual_array),
                "mae": round(float(np.mean(np.abs(errors))), 2),
                "rmse": round(float(np.sqrt(np.mean(errors**2))), 2),
                "mape": round(_safe_mape(actual_array, predicted_array), 2),
                "selected_default": model_name == "linear_trend_with_seasonal_anchor",
                "notes": (
                    "Default reporting model because it is transparent and explainable."
                    if model_name == "linear_trend_with_seasonal_anchor"
                    else "Comparison model for benchmarking forecast accuracy."
                ),
            }
        )
    if not random_forest_available:
        rows.append(
            {
                "model": "random_forest_regressor",
                "test_periods": 0,
                "mae": None,
                "rmse": None,
                "mape": None,
                "selected_default": False,
                "notes": "Install scikit-learn to run this optional benchmark.",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def detect_reporting_anomalies(facility_monthly: pd.DataFrame) -> pd.DataFrame:
    """Detect unusual hospital-month submission changes using rolling z scores."""

    if facility_monthly.empty:
        return pd.DataFrame(columns=list(facility_monthly.columns) + ["rolling_mean", "rolling_std", "z_score", "pct_change", "anomaly_flag", "anomaly_reason"])

    frame = facility_monthly.copy()
    frame["month_dt"] = pd.to_datetime(frame["month"])
    frame = frame.sort_values(["hospital_id", "month_dt"]).reset_index(drop=True)

    def add_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        shifted = group["total_cases"].shift(1)
        group["rolling_mean"] = shifted.rolling(window=6, min_periods=3).mean()
        group["rolling_std"] = shifted.rolling(window=6, min_periods=3).std().replace(0, np.nan)
        group["z_score"] = (group["total_cases"] - group["rolling_mean"]) / group["rolling_std"]
        group["pct_change"] = group["total_cases"].pct_change().replace([np.inf, -np.inf], np.nan)
        return group

    frame = pd.concat(
        [add_features(group) for _, group in frame.groupby("hospital_id", sort=False)],
        ignore_index=True,
    )
    frame["z_score"] = frame["z_score"].fillna(0).round(2)
    frame["pct_change"] = (frame["pct_change"].fillna(0) * 100).round(2)
    frame["rolling_mean"] = frame["rolling_mean"].fillna(frame["total_cases"]).round(2)
    frame["rolling_std"] = frame["rolling_std"].fillna(0).round(2)
    frame["anomaly_flag"] = (frame["z_score"].abs() >= 2.5) | (frame["pct_change"].abs() >= 55)
    frame["anomaly_reason"] = np.select(
        [
            frame["z_score"] >= 2.5,
            frame["z_score"] <= -2.5,
            frame["pct_change"] >= 55,
            frame["pct_change"] <= -55,
        ],
        [
            "High volume versus rolling baseline",
            "Low volume versus rolling baseline",
            "Large month-over-month increase",
            "Large month-over-month decrease",
        ],
        default="Within expected range",
    )
    frame = frame.drop(columns=["month_dt"])
    return frame
