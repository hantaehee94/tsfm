from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from chronos2_core import run_prediction


WEEKDAY_COVARIATE_COLUMN = "__weekday__"
WEEKDAY_CODE_COVARIATE_COLUMN = "__weekday_code__"
WEEKDAY_SIN_COVARIATE_COLUMN = "__weekday_sin__"
WEEKDAY_COS_COVARIATE_COLUMN = "__weekday_cos__"
WEEKDAY_ENCODINGS = ("문자열", "0~6", "sin/cos")
WEEKDAY_ENCODING_LABELS = {"문자열": "文字", "0~6": "0~6", "sin/cos": "sin.cos"}
WEEKDAY_COVARIATE_SCOPES = (("過去", False), ("過去+未来", True))
WEEKDAY_BASELINE_SCENARIO = "曜日-無"


def build_weekday_scenario_name(encoding: str, include_future_covariate: bool) -> str:
    encoding_label = WEEKDAY_ENCODING_LABELS.get(encoding, encoding)
    scope_label = "過去+未来" if include_future_covariate else "過去"
    return f"曜日-有({encoding_label},{scope_label})"


def get_weekday_scenario_color(scenario: str) -> str:
    line_colors = {
        WEEKDAY_BASELINE_SCENARIO: "#1d3557",
        build_weekday_scenario_name("문자열", False): "#e76f51",
        build_weekday_scenario_name("문자열", True): "#f4a261",
        build_weekday_scenario_name("0~6", False): "#457b9d",
        build_weekday_scenario_name("0~6", True): "#2a9d8f",
        build_weekday_scenario_name("sin/cos", False): "#7b2cbf",
        build_weekday_scenario_name("sin/cos", True): "#6c757d",
    }
    return line_colors.get(scenario, "#495057")


def add_weekday_covariate(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    include_future_covariate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    prepared_context = context_df.copy()
    prepared_context[timestamp_column] = pd.to_datetime(prepared_context[timestamp_column], errors="coerce")
    prepared_context[WEEKDAY_COVARIATE_COLUMN] = prepared_context[timestamp_column].dt.day_name()

    if not include_future_covariate:
        return prepared_context, future_df.copy() if future_df is not None else future_df

    if future_df is not None and not future_df.empty:
        prepared_future = future_df.copy()
        prepared_future[timestamp_column] = pd.to_datetime(prepared_future[timestamp_column], errors="coerce")
        prepared_future[WEEKDAY_COVARIATE_COLUMN] = prepared_future[timestamp_column].dt.day_name()
        return prepared_context, prepared_future

    derived_future = prepared_context[[id_column, timestamp_column]].drop_duplicates().copy()
    derived_future[WEEKDAY_COVARIATE_COLUMN] = derived_future[timestamp_column].dt.day_name()
    return prepared_context, derived_future


def add_weekday_code_covariate(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    include_future_covariate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    prepared_context = context_df.copy()
    prepared_context[timestamp_column] = pd.to_datetime(prepared_context[timestamp_column], errors="coerce")
    prepared_context[WEEKDAY_CODE_COVARIATE_COLUMN] = prepared_context[timestamp_column].dt.dayofweek.astype("float64")

    if not include_future_covariate:
        return prepared_context, future_df.copy() if future_df is not None else future_df

    if future_df is not None and not future_df.empty:
        prepared_future = future_df.copy()
        prepared_future[timestamp_column] = pd.to_datetime(prepared_future[timestamp_column], errors="coerce")
        prepared_future[WEEKDAY_CODE_COVARIATE_COLUMN] = prepared_future[timestamp_column].dt.dayofweek.astype("float64")
        return prepared_context, prepared_future

    derived_future = prepared_context[[id_column, timestamp_column]].drop_duplicates().copy()
    derived_future[WEEKDAY_CODE_COVARIATE_COLUMN] = derived_future[timestamp_column].dt.dayofweek.astype("float64")
    return prepared_context, derived_future


def add_weekday_cyclical_covariates(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    include_future_covariate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    def _append_weekday_cyclical_columns(df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
        weekday_code = prepared[timestamp_column].dt.dayofweek.astype("float64")
        angle = 2.0 * np.pi * weekday_code / 7.0
        prepared[WEEKDAY_SIN_COVARIATE_COLUMN] = np.sin(angle)
        prepared[WEEKDAY_COS_COVARIATE_COLUMN] = np.cos(angle)
        return prepared

    prepared_context = _append_weekday_cyclical_columns(context_df)

    if not include_future_covariate:
        return prepared_context, future_df.copy() if future_df is not None else future_df

    if future_df is not None and not future_df.empty:
        prepared_future = _append_weekday_cyclical_columns(future_df)
        return prepared_context, prepared_future

    derived_future = prepared_context[[id_column, timestamp_column]].drop_duplicates().copy()
    derived_future = _append_weekday_cyclical_columns(derived_future)
    return prepared_context, derived_future


def build_weekday_covariate_frames(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    encoding: str,
    include_future_covariate: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    builders = {
        "문자열": add_weekday_covariate,
        "0~6": add_weekday_code_covariate,
        "sin/cos": add_weekday_cyclical_covariates,
    }
    scenario_builder = builders[encoding]
    return scenario_builder(
        context_df=context_df,
        future_df=future_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        include_future_covariate=include_future_covariate,
    )


def infer_series_frequency(timestamps: pd.Series) -> pd.Timedelta:
    prepared = pd.to_datetime(timestamps, errors="coerce").dropna().sort_values()
    if len(prepared) < 2:
        raise ValueError("미래 요일 공변량을 만들려면 시계열당 최소 2개 timestamp가 필요합니다.")

    deltas = prepared.diff().dropna()
    positive_deltas = deltas.loc[deltas > pd.Timedelta(0)]
    if positive_deltas.empty:
        raise ValueError("timestamp 간격을 추정하지 못했습니다.")
    return positive_deltas.value_counts().idxmax()


def build_future_calendar_frame(
    context_df: pd.DataFrame,
    actual_future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    prediction_length: int,
) -> pd.DataFrame:
    if actual_future_df is not None and not actual_future_df.empty:
        prepared_actual = actual_future_df[[id_column, timestamp_column]].drop_duplicates().copy()
        prepared_actual[timestamp_column] = pd.to_datetime(prepared_actual[timestamp_column], errors="coerce")
        return prepared_actual

    prepared_context = context_df.copy()
    prepared_context[timestamp_column] = pd.to_datetime(prepared_context[timestamp_column], errors="coerce")
    prepared_context = prepared_context.sort_values([id_column, timestamp_column])

    rows: list[dict[str, object]] = []
    for series_id, group in prepared_context.groupby(id_column, dropna=False):
        valid_timestamps = group[timestamp_column].dropna().sort_values()
        if valid_timestamps.empty:
            raise ValueError(f"`{series_id}` 시계열의 timestamp가 비어 있습니다.")

        frequency = infer_series_frequency(valid_timestamps)
        last_timestamp = valid_timestamps.iloc[-1]
        for horizon in range(1, prediction_length + 1):
            rows.append(
                {
                    id_column: series_id,
                    timestamp_column: last_timestamp + frequency * horizon,
                }
            )

    return pd.DataFrame(rows)


def prepare_weekday_forecast_inputs(
    model_context_df: pd.DataFrame,
    model_future_df: pd.DataFrame | None,
    actual_future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    prediction_length: int,
    encoding: str,
    include_future_covariate: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    scenario_future_df = model_future_df
    if include_future_covariate and (scenario_future_df is None or scenario_future_df.empty):
        scenario_future_df = build_future_calendar_frame(
            context_df=model_context_df,
            actual_future_df=actual_future_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
            prediction_length=prediction_length,
        )

    return build_weekday_covariate_frames(
        context_df=model_context_df,
        future_df=scenario_future_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        encoding=encoding,
        include_future_covariate=include_future_covariate,
    )


def run_forecast_weekday_comparison(
    pipeline,
    baseline_pred_df: pd.DataFrame,
    model_context_df: pd.DataFrame,
    model_future_df: pd.DataFrame | None,
    actual_future_df: pd.DataFrame | None,
    prediction_length: int,
    id_column: str,
    timestamp_column: str,
    target_column: str,
    metrics_builder: Callable[..., pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    error_rows: list[dict[str, object]] = []

    def append_prediction(
        pred_df: pd.DataFrame,
        scenario: str,
        weekday_encoding: str,
        weekday_scope: str,
    ) -> None:
        prepared_pred = pred_df.copy()
        prepared_pred["scenario"] = scenario
        prepared_pred["weekday_encoding"] = weekday_encoding
        prepared_pred["weekday_scope"] = weekday_scope
        pred_frames.append(prepared_pred)

        if actual_future_df is None or actual_future_df.empty:
            return
        if metrics_builder is None:
            raise ValueError("실측 미래 구간이 있을 때는 metrics_builder가 필요합니다.")

        metrics = metrics_builder(
            pred_df=pred_df,
            history_df=model_context_df,
            actual_df=actual_future_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_column=target_column,
        )
        metrics["scenario"] = scenario
        metrics["weekday_encoding"] = weekday_encoding
        metrics["weekday_scope"] = weekday_scope
        metric_frames.append(metrics)

    append_prediction(
        pred_df=baseline_pred_df,
        scenario=WEEKDAY_BASELINE_SCENARIO,
        weekday_encoding="無",
        weekday_scope="無",
    )

    for encoding in WEEKDAY_ENCODINGS:
        for scope_name, include_future_covariate in WEEKDAY_COVARIATE_SCOPES:
            scenario_name = build_weekday_scenario_name(encoding, include_future_covariate)
            try:
                scenario_context_df, scenario_future_df = prepare_weekday_forecast_inputs(
                    model_context_df=model_context_df,
                    model_future_df=model_future_df,
                    actual_future_df=actual_future_df,
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                    prediction_length=prediction_length,
                    encoding=encoding,
                    include_future_covariate=include_future_covariate,
                )
                scenario_pred_df = run_prediction(
                    pipeline=pipeline,
                    context_df=scenario_context_df,
                    future_df=scenario_future_df,
                    prediction_length=prediction_length,
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                    target_column=target_column,
                )
                append_prediction(
                    pred_df=scenario_pred_df,
                    scenario=scenario_name,
                    weekday_encoding=WEEKDAY_ENCODING_LABELS.get(encoding, encoding),
                    weekday_scope=scope_name,
                )
            except Exception as exc:
                error_rows.append(
                    {
                        "scenario": scenario_name,
                        "weekday_encoding": WEEKDAY_ENCODING_LABELS.get(encoding, encoding),
                        "weekday_scope": scope_name,
                        "error": str(exc),
                    }
                )

    comparison_pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    comparison_metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    comparison_errors_df = pd.DataFrame(error_rows)
    return comparison_pred_df, comparison_metrics_df, comparison_errors_df


def build_forecast_weekday_metric_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or "scenario" not in metrics_df.columns:
        return metrics_df

    metric_columns = ["MAE", "RMSE", "MAPE", "MASE", "Correlation"]
    available_metrics = [col for col in metric_columns if col in metrics_df.columns]
    summary_df = metrics_df[["scenario", "weekday_encoding", "weekday_scope", *available_metrics]].copy()
    baseline_rows = summary_df.loc[summary_df["scenario"] == WEEKDAY_BASELINE_SCENARIO]
    if baseline_rows.empty:
        return summary_df

    baseline = baseline_rows.iloc[0]
    for metric in available_metrics:
        summary_df[f"{metric}_delta"] = summary_df[metric] - baseline[metric]
    return summary_df
