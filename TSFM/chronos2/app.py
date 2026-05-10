from __future__ import annotations

from io import BytesIO
from math import ceil

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chronos2_core import (
    CHRONOS2_MAX_CONTEXT_LENGTH,
    CHRONOS2_MAX_PREDICTION_LENGTH,
    detect_device,
    load_pipeline,
    load_table,
    run_prediction,
    trim_context_to_model_limit,
)


st.set_page_config(page_title="Chronos-2 Local Lab", layout="wide")
st.title("Chronos-2 Local Lab")
st.caption("맥북에서 로컬로 Chronos-2 예측 실험을 빠르게 돌리기 위한 최소 GUI")

AUTO_ID_OPTION = "__single_series__"
AUTO_ID_COLUMN = "__auto_id__"
WEEKDAY_COVARIATE_COLUMN = "__weekday__"
WEEKDAY_CODE_COVARIATE_COLUMN = "__weekday_code__"
WEEKDAY_SIN_COVARIATE_COLUMN = "__weekday_sin__"
WEEKDAY_COS_COVARIATE_COLUMN = "__weekday_cos__"
WEEKDAY_ENCODINGS = ("문자열", "0~6", "sin/cos")
WEEKDAY_ENCODING_LABELS = {"문자열": "文字", "0~6": "0~6", "sin/cos": "sin.cos"}
WEEKDAY_COVARIATE_SCOPES = (("過去", False), ("過去+未来", True))
WEEKDAY_BASELINE_SCENARIO = "曜日-無"
POINT_PREDICTION_COLUMN_CANDIDATES = ("predictions", 0.5, "0.5")
INFLUENCE_SOURCE_CONTEXT = "과거 context"
INFLUENCE_SOURCE_FUTURE = "미래 known covariate"


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


@st.cache_resource(show_spinner=False)
def get_pipeline(model_id: str, device: str):
    return load_pipeline(model_id, device)


@st.cache_data(show_spinner=False)
def load_uploaded_table(uploaded_file_name: str, raw_bytes: bytes) -> pd.DataFrame:
    return load_table(uploaded_file_name, raw_bytes)


def guess_id_column(df: pd.DataFrame) -> str:
    preferred_names = ["id", "item_id", "series_id", "unique_id"]
    lower_map = {col.lower(): col for col in df.columns}
    for name in preferred_names:
        if name in lower_map:
            return lower_map[name]
    return AUTO_ID_OPTION


def guess_timestamp_column(df: pd.DataFrame) -> str:
    preferred_names = ["timestamp", "date", "datetime", "ds", "time"]
    lower_map = {col.lower(): col for col in df.columns}
    for name in preferred_names:
        if name in lower_map:
            return lower_map[name]
    return df.columns[0]


def guess_target_column(df: pd.DataFrame) -> str:
    preferred_names = ["target", "y", "value", "values"]
    lower_map = {col.lower(): col for col in df.columns}
    for name in preferred_names:
        if name in lower_map:
            return lower_map[name]

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        return numeric_columns[0]
    return df.columns[-1]


def trim_to_recent_history(
    context_df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
    history_limit: int | None,
) -> pd.DataFrame:
    prepared = context_df.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column])
    prepared = prepared.sort_values([id_column, timestamp_column])
    if history_limit is None:
        return prepared
    return prepared.groupby(id_column, group_keys=False).tail(history_limit)


def apply_id_selection(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_selection: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    prepared_context = context_df.copy()
    prepared_future = future_df.copy() if future_df is not None else None

    if id_selection == AUTO_ID_OPTION:
        prepared_context[AUTO_ID_COLUMN] = "series_1"
        if prepared_future is not None:
            prepared_future[AUTO_ID_COLUMN] = "series_1"
        return prepared_context, prepared_future, AUTO_ID_COLUMN

    return prepared_context, prepared_future, id_selection


def filter_model_columns(
    context_df: pd.DataFrame,
    full_context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    target_column: str,
    past_only_covariates: list[str],
    known_future_covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    context_covariates = list(dict.fromkeys([*past_only_covariates, *known_future_covariates]))
    keep_context_columns = [id_column, timestamp_column, target_column, *context_covariates]
    prepared_context = context_df.loc[:, [col for col in keep_context_columns if col in context_df.columns]].copy()
    prepared_full_context = full_context_df.loc[:, [col for col in keep_context_columns if col in full_context_df.columns]].copy()

    if future_df is None or future_df.empty:
        return prepared_context, prepared_full_context, future_df

    keep_future_columns = [id_column, timestamp_column, *known_future_covariates]
    prepared_future = future_df.loc[:, keep_future_columns].copy()
    return prepared_context, prepared_full_context, prepared_future


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


def trim_by_index_window(
    df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    start_idx: int,
    end_idx: int,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df

    prepared = df.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    prepared = prepared.sort_values([id_column, timestamp_column])
    positions = prepared.groupby(id_column).cumcount()
    prepared = prepared.loc[(positions >= start_idx) & (positions <= end_idx)].copy()
    return prepared


def get_series_lengths(df: pd.DataFrame, id_column: str, timestamp_column: str) -> pd.Series:
    prepared = df.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    prepared = prepared.sort_values([id_column, timestamp_column])
    return prepared.groupby(id_column).size()


def get_index_timestamp_map(
    df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
    selected_id: str,
) -> pd.Series:
    prepared = df.loc[df[id_column].astype(str) == selected_id].copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    prepared = prepared.sort_values(timestamp_column).reset_index(drop=True)
    return prepared[timestamp_column]


def build_evaluation_split(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    prediction_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    prepared = context_df.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    prepared = prepared.sort_values([id_column, timestamp_column])

    lengths = prepared.groupby(id_column).size()
    if lengths.min() < prediction_length + 3:
        raise ValueError(
            f"선택한 구간이 너무 짧습니다. 각 시계열은 최소 {prediction_length + 3}개 이상이어야 합니다."
        )

    positions = prepared.groupby(id_column).cumcount()
    split_points = lengths.reindex(prepared[id_column]).to_numpy() - prediction_length
    model_context = prepared.loc[positions < split_points].copy()
    actual_future = prepared.loc[positions >= split_points].copy()

    if future_df is None or future_df.empty:
        return model_context, actual_future, None

    prepared_future = future_df.copy()
    prepared_future[timestamp_column] = pd.to_datetime(prepared_future[timestamp_column], errors="coerce")
    horizon_keys = actual_future[[id_column, timestamp_column]].drop_duplicates()
    filtered_future = prepared_future.merge(horizon_keys, on=[id_column, timestamp_column], how="inner")
    return model_context, actual_future, filtered_future


def build_future_comparison_split(
    full_context_df: pd.DataFrame,
    model_context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    prediction_length: int,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    prepared_full = full_context_df.copy()
    prepared_full[timestamp_column] = pd.to_datetime(prepared_full[timestamp_column], errors="coerce")
    prepared_full = prepared_full.sort_values([id_column, timestamp_column])

    prepared_context = model_context_df.copy()
    prepared_context[timestamp_column] = pd.to_datetime(prepared_context[timestamp_column], errors="coerce")
    prepared_context = prepared_context.sort_values([id_column, timestamp_column])

    context_lengths = prepared_context.groupby(id_column).size()
    full_positions = prepared_full.groupby(id_column).cumcount()
    horizon_start = context_lengths.reindex(prepared_full[id_column]).to_numpy()
    horizon_end = horizon_start + prediction_length
    actual_future = prepared_full.loc[(full_positions >= horizon_start) & (full_positions < horizon_end)].copy()

    if actual_future.empty:
        if future_df is not None and not future_df.empty:
            prepared_future = future_df.copy()
            prepared_future[timestamp_column] = pd.to_datetime(prepared_future[timestamp_column], errors="coerce")
            return None, prepared_future
        return None, None

    if future_df is None or future_df.empty:
        return actual_future, None

    prepared_future = future_df.copy()
    prepared_future[timestamp_column] = pd.to_datetime(prepared_future[timestamp_column], errors="coerce")
    horizon_keys = actual_future[[id_column, timestamp_column]].drop_duplicates()
    filtered_future = prepared_future.merge(horizon_keys, on=[id_column, timestamp_column], how="inner")
    return actual_future, filtered_future


def prepare_model_inputs_for_forecast(
    context_df: pd.DataFrame,
    full_context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    prediction_length: int,
    experiment_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    model_context_df = trim_context_to_model_limit(
        context_df=context_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    model_future_df = future_df
    actual_future_df = None

    if experiment_mode == "데이터셋 내부 평가":
        model_context_df, actual_future_df, model_future_df = build_evaluation_split(
            context_df=context_df,
            future_df=future_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
            prediction_length=prediction_length,
        )
    elif experiment_mode == "선택 구간 끝에서 미래 예측":
        actual_future_df, model_future_df = build_future_comparison_split(
            full_context_df=full_context_df,
            model_context_df=model_context_df,
            future_df=future_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
            prediction_length=prediction_length,
        )

    model_context_df = trim_context_to_model_limit(
        context_df=model_context_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    return model_context_df, model_future_df, actual_future_df


def compute_metrics(
    pred_df: pd.DataFrame,
    history_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
    target_column: str,
) -> pd.DataFrame:
    point_column = "predictions" if "predictions" in pred_df.columns else 0.5 if 0.5 in pred_df.columns else "0.5"
    history_prepared = history_df[[id_column, timestamp_column, target_column]].copy()
    history_prepared[timestamp_column] = pd.to_datetime(history_prepared[timestamp_column], errors="coerce")
    history_prepared = history_prepared.sort_values([id_column, timestamp_column])

    actual_prepared = actual_df[[id_column, timestamp_column, target_column]].copy()
    actual_prepared = actual_prepared.rename(columns={target_column: "actual"})
    merged = pred_df.merge(actual_prepared, on=[id_column, timestamp_column], how="inner")
    merged["abs_error"] = (merged[point_column] - merged["actual"]).abs()
    merged["sq_error"] = (merged[point_column] - merged["actual"]) ** 2
    merged["ape"] = np.where(merged["actual"].abs() > 1e-8, merged["abs_error"] / merged["actual"].abs(), np.nan)

    history_prepared["naive_abs_error"] = history_prepared.groupby(id_column)[target_column].diff().abs()
    mase_scale = (
        history_prepared.groupby(id_column, dropna=False)["naive_abs_error"]
        .mean()
        .rename("mase_scale")
        .reset_index()
    )
    merged = merged.merge(mase_scale, on=id_column, how="left")
    merged["scaled_abs_error"] = np.where(
        merged["mase_scale"] > 1e-8,
        merged["abs_error"] / merged["mase_scale"],
        np.nan,
    )

    correlations: list[float] = []
    for _, group in merged.groupby(id_column, dropna=False):
        if len(group) < 2:
            continue
        pred_std = float(group[point_column].std(ddof=0))
        actual_std = float(group["actual"].std(ddof=0))
        if pred_std <= 1e-8 or actual_std <= 1e-8:
            continue
        corr = group[point_column].corr(group["actual"])
        if pd.notna(corr):
            correlations.append(float(corr))

    return pd.DataFrame(
        {
            "MAE": [float(merged["abs_error"].mean())],
            "RMSE": [float(np.sqrt(merged["sq_error"].mean()))],
            "MAPE": [float(np.nanmean(merged["ape"]) * 100.0)],
            "MASE": [float(np.nanmean(merged["scaled_abs_error"]))],
            "Correlation": [float(np.nanmean(correlations)) if correlations else np.nan],
        }
    )


def downsample_frame(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = ceil(len(df) / max_points)
    return df.iloc[::step].copy()


def build_plot_frame(
    history_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    actual_future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    target_column: str,
    selected_id: str,
) -> go.Figure:
    history = history_df.loc[history_df[id_column].astype(str) == selected_id, [timestamp_column, target_column]].copy()
    history[timestamp_column] = pd.to_datetime(history[timestamp_column], errors="coerce")
    history = history.sort_values(timestamp_column)

    forecast_columns = [col for col in ["predictions", 0.1, 0.5, 0.9, "0.1", "0.5", "0.9"] if col in pred_df.columns]
    forecast = pred_df.loc[pred_df[id_column].astype(str) == selected_id, [timestamp_column, *forecast_columns]].copy()
    forecast[timestamp_column] = pd.to_datetime(forecast[timestamp_column], errors="coerce")
    forecast = forecast.sort_values(timestamp_column)

    actual_future = None
    if actual_future_df is not None and not actual_future_df.empty:
        actual_future = actual_future_df.loc[
            actual_future_df[id_column].astype(str) == selected_id,
            [timestamp_column, target_column],
        ].copy()
        actual_future[timestamp_column] = pd.to_datetime(actual_future[timestamp_column], errors="coerce")
        actual_future = actual_future.sort_values(timestamp_column)

    history = downsample_frame(history)
    forecast = downsample_frame(forecast)
    if actual_future is not None:
        actual_future = downsample_frame(actual_future)

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=history[timestamp_column],
            y=history[target_column],
            mode="lines",
            name="history",
            line={"color": "#264653", "width": 2},
        )
    )

    if actual_future is not None:
        figure.add_trace(
            go.Scattergl(
                x=actual_future[timestamp_column],
                y=actual_future[target_column],
                mode="lines",
                name="actual_future",
                line={"color": "#2a9d8f", "width": 2},
            )
        )

    point_column = "predictions" if "predictions" in forecast.columns else 0.5 if 0.5 in forecast.columns else "0.5"
    figure.add_trace(
        go.Scattergl(
            x=forecast[timestamp_column],
            y=forecast[point_column],
            mode="lines",
            name="prediction",
            line={"color": "#e76f51", "width": 2},
        )
    )

    for column, color in [(0.1, "#f4a261"), ("0.1", "#f4a261"), (0.9, "#f4a261"), ("0.9", "#f4a261")]:
        if column in forecast.columns:
            figure.add_trace(
                go.Scattergl(
                    x=forecast[timestamp_column],
                    y=forecast[column],
                    mode="lines",
                    name=f"quantile_{column}",
                    line={"color": color, "width": 1, "dash": "dot"},
                )
            )

    figure.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h"},
    )
    return figure


def show_context_summary(
    context_df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
) -> None:
    if context_df.empty or not id_column or not timestamp_column:
        return

    prepared = context_df.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    lengths = prepared.groupby(id_column).size()
    if lengths.empty:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("시계열 개수", int(lengths.shape[0]))
    col2.metric("최소 과거 길이", int(lengths.min()))
    col3.metric("최대 과거 길이", int(lengths.max()))

    if lengths.min() < 3:
        shortest_id = str(lengths.idxmin())
        st.error(
            "현재 선택한 `id` 컬럼 기준으로 어떤 시계열은 길이가 3 미만입니다. "
            f"가장 짧은 시계열은 `{shortest_id}`이고 길이는 {int(lengths.min())}입니다. "
            "보통 `id` 컬럼이 잘못 선택된 경우 이런 문제가 생깁니다."
        )


def show_series_preview(
    context_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    actual_future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    target_column: str,
) -> None:
    if pred_df.empty:
        return

    available_ids = context_df[id_column].astype(str).unique().tolist()
    selected_id = st.selectbox("미리볼 시계열", options=available_ids)

    figure = build_plot_frame(
        history_df=context_df,
        pred_df=pred_df,
        actual_future_df=actual_future_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_column=target_column,
        selected_id=selected_id,
    )
    st.plotly_chart(figure, use_container_width=True)


def save_predictions_download(pred_df: pd.DataFrame) -> None:
    buffer = BytesIO()
    pred_df.to_csv(buffer, index=False)
    st.download_button(
        label="예측 결과 CSV 다운로드",
        data=buffer.getvalue(),
        file_name="chronos2_predictions.csv",
        mime="text/csv",
    )


def show_model_input_summary(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    target_column: str,
) -> None:
    st.markdown("**모델 입력 요약**")

    context_lengths = get_series_lengths(
        context_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    context_covariates = [
        col for col in context_df.columns if col not in {id_column, timestamp_column, target_column}
    ]

    info1, info2, info3, info4 = st.columns(4)
    info1.metric("입력 시계열 개수", int(context_lengths.shape[0]))
    info2.metric("최대 history 길이", int(context_lengths.max()))
    info3.metric("최소 history 길이", int(context_lengths.min()))
    info4.metric("과거 covariates 수", int(len(context_covariates)))

    st.caption(
        "아래 표는 Chronos-2에 실제로 전달되는 입력입니다. "
        f"history는 시계열당 최근 최대 {CHRONOS2_MAX_CONTEXT_LENGTH}개 시점만 사용됩니다."
    )

    summary_rows = [
        {
            "input": "context_df",
            "rows": int(len(context_df)),
            "columns": ", ".join(context_df.columns.astype(str).tolist()),
            "covariates": ", ".join(context_covariates) if context_covariates else "(없음)",
        }
    ]

    if future_df is None or future_df.empty:
        summary_rows.append(
            {
                "input": "future_df",
                "rows": 0,
                "columns": "(없음)",
                "covariates": "(없음)",
            }
        )
    else:
        future_covariates = [
            col for col in future_df.columns if col not in {id_column, timestamp_column, target_column}
        ]
        summary_rows.append(
            {
                "input": "future_df",
                "rows": int(len(future_df)),
                "columns": ", ".join(future_df.columns.astype(str).tolist()),
                "covariates": ", ".join(future_covariates) if future_covariates else "(없음)",
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


def get_point_prediction_column(
    pred_df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
) -> object:
    for column in POINT_PREDICTION_COLUMN_CANDIDATES:
        if column in pred_df.columns:
            return column

    excluded_columns = {id_column, timestamp_column}
    numeric_columns = [
        col for col in pred_df.select_dtypes(include=["number"]).columns if col not in excluded_columns
    ]
    if numeric_columns:
        return numeric_columns[0]
    raise ValueError("예측 결과에서 point prediction 컬럼을 찾지 못했습니다.")


def extract_selected_point_forecast(
    pred_df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
    selected_id: str,
) -> pd.DataFrame:
    point_column = get_point_prediction_column(
        pred_df=pred_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    forecast = pred_df.loc[
        pred_df[id_column].astype(str) == selected_id,
        [timestamp_column, point_column],
    ].copy()
    if forecast.empty:
        raise ValueError(f"`{selected_id}` 시계열의 예측 결과를 찾지 못했습니다.")

    forecast[timestamp_column] = pd.to_datetime(forecast[timestamp_column], errors="coerce")
    forecast = forecast.sort_values(timestamp_column).reset_index(drop=True)
    forecast = forecast.rename(columns={point_column: "prediction"})
    forecast["horizon_index"] = np.arange(1, len(forecast) + 1)
    forecast["horizon_label"] = forecast["horizon_index"].map(lambda value: f"h+{int(value)}")
    return forecast


def format_short_timestamp(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    if timestamp.hour == 0 and timestamp.minute == 0 and timestamp.second == 0:
        return timestamp.strftime("%Y-%m-%d")
    return timestamp.strftime("%Y-%m-%d %H:%M")


def get_series_rows_for_influence(
    df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
    selected_id: str,
) -> pd.DataFrame:
    series = df.loc[df[id_column].astype(str) == selected_id].copy()
    if series.empty:
        raise ValueError(f"`{selected_id}` 시계열을 찾지 못했습니다.")

    series[timestamp_column] = pd.to_datetime(series[timestamp_column], errors="coerce")
    return series.sort_values(timestamp_column).reset_index(drop=False)


def build_influence_windows(
    source_df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
    selected_id: str,
    input_source: str,
    analysis_length: int,
    window_size: int,
    stride: int,
    max_windows: int,
) -> list[dict[str, object]]:
    series = get_series_rows_for_influence(
        df=source_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        selected_id=selected_id,
    )
    total_length = len(series)
    if total_length == 0:
        return []

    analysis_length = min(max(1, int(analysis_length)), total_length)
    window_size = min(max(1, int(window_size)), analysis_length)
    stride = max(1, int(stride))
    recent_start = max(0, total_length - analysis_length)
    prefix = "hist" if input_source == INFLUENCE_SOURCE_CONTEXT else "future"

    windows: list[dict[str, object]] = []
    start_idx = recent_start
    while start_idx < total_length:
        end_idx = min(total_length - 1, start_idx + window_size - 1)
        window_rows = series.iloc[start_idx : end_idx + 1]
        window_start = window_rows[timestamp_column].iloc[0]
        window_end = window_rows[timestamp_column].iloc[-1]
        windows.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "window_label": f"{prefix} {int(start_idx)}-{int(end_idx)}",
                "window_time_range": (
                    f"{format_short_timestamp(window_start)} ~ {format_short_timestamp(window_end)}"
                ),
                "window_start": window_start,
                "window_end": window_end,
            }
        )
        if end_idx >= total_length - 1:
            break
        start_idx += stride

    if max_windows > 0 and len(windows) > max_windows:
        return windows[-max_windows:]
    return windows


def get_perturbation_replacement(values: pd.Series, replacement_method: str) -> object:
    if replacement_method == "zero" and pd.api.types.is_numeric_dtype(values):
        return 0.0

    clean_values = values.dropna()
    if pd.api.types.is_numeric_dtype(values):
        if clean_values.empty:
            return 0.0
        return float(clean_values.mean())

    if clean_values.empty:
        return ""
    modes = clean_values.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return clean_values.iloc[0]


def perturb_series_window(
    df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
    selected_id: str,
    perturb_column: str,
    start_idx: int,
    end_idx: int,
    replacement_method: str,
) -> pd.DataFrame:
    if perturb_column not in df.columns:
        raise ValueError(f"`{perturb_column}` 컬럼을 찾지 못했습니다.")

    prepared = df.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    selected_mask = prepared[id_column].astype(str) == selected_id
    ordered_indices = prepared.loc[selected_mask].sort_values(timestamp_column).index.tolist()
    target_indices = ordered_indices[int(start_idx) : int(end_idx) + 1]
    if not target_indices:
        return prepared

    replacement = get_perturbation_replacement(
        values=prepared.loc[selected_mask, perturb_column],
        replacement_method=replacement_method,
    )
    prepared.loc[target_indices, perturb_column] = replacement
    return prepared


def run_prediction_influence_analysis(
    pipeline,
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    prediction_length: int,
    id_column: str,
    timestamp_column: str,
    target_column: str,
    selected_id: str,
    input_source: str,
    perturb_column: str,
    analysis_length: int,
    window_size: int,
    stride: int,
    max_windows: int,
    replacement_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_pred_df = run_prediction(
        pipeline=pipeline,
        context_df=context_df,
        future_df=future_df,
        prediction_length=prediction_length,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_column=target_column,
    )
    baseline_forecast = extract_selected_point_forecast(
        pred_df=baseline_pred_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        selected_id=selected_id,
    ).rename(columns={"prediction": "baseline_prediction"})

    if input_source == INFLUENCE_SOURCE_CONTEXT:
        source_df = context_df
    elif future_df is not None and not future_df.empty:
        source_df = future_df
    else:
        raise ValueError("미래 공변량 입력이 없습니다.")

    windows = build_influence_windows(
        source_df=source_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        selected_id=selected_id,
        input_source=input_source,
        analysis_length=analysis_length,
        window_size=window_size,
        stride=stride,
        max_windows=max_windows,
    )
    if not windows:
        raise ValueError("영향도 분석에 사용할 window가 없습니다.")

    influence_rows: list[dict[str, object]] = []
    for window in windows:
        perturbed_context_df = context_df
        perturbed_future_df = future_df
        if input_source == INFLUENCE_SOURCE_CONTEXT:
            perturbed_context_df = perturb_series_window(
                df=context_df,
                id_column=id_column,
                timestamp_column=timestamp_column,
                selected_id=selected_id,
                perturb_column=perturb_column,
                start_idx=int(window["start_idx"]),
                end_idx=int(window["end_idx"]),
                replacement_method=replacement_method,
            )
        else:
            perturbed_future_df = perturb_series_window(
                df=future_df if future_df is not None else pd.DataFrame(),
                id_column=id_column,
                timestamp_column=timestamp_column,
                selected_id=selected_id,
                perturb_column=perturb_column,
                start_idx=int(window["start_idx"]),
                end_idx=int(window["end_idx"]),
                replacement_method=replacement_method,
            )

        perturbed_pred_df = run_prediction(
            pipeline=pipeline,
            context_df=perturbed_context_df,
            future_df=perturbed_future_df,
            prediction_length=prediction_length,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_column=target_column,
        )
        perturbed_forecast = extract_selected_point_forecast(
            pred_df=perturbed_pred_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
            selected_id=selected_id,
        ).rename(columns={"prediction": "perturbed_prediction"})

        merged = baseline_forecast.merge(
            perturbed_forecast[[timestamp_column, "perturbed_prediction"]],
            on=timestamp_column,
            how="inner",
        )
        for _, row in merged.iterrows():
            delta = float(row["perturbed_prediction"] - row["baseline_prediction"])
            influence_rows.append(
                {
                    "input_source": input_source,
                    "input_column": perturb_column,
                    "selected_id": selected_id,
                    "window_label": window["window_label"],
                    "window_time_range": window["window_time_range"],
                    "start_idx": int(window["start_idx"]),
                    "end_idx": int(window["end_idx"]),
                    "horizon_index": int(row["horizon_index"]),
                    "horizon_label": row["horizon_label"],
                    "forecast_timestamp": row[timestamp_column],
                    "baseline_prediction": float(row["baseline_prediction"]),
                    "perturbed_prediction": float(row["perturbed_prediction"]),
                    "delta": delta,
                    "abs_delta": abs(delta),
                }
            )

    if not influence_rows:
        raise ValueError("baseline과 perturb 예측의 timestamp가 맞지 않아 influence를 계산하지 못했습니다.")

    return pd.DataFrame(influence_rows), baseline_forecast


def summarize_influence_windows(influence_df: pd.DataFrame) -> pd.DataFrame:
    if influence_df.empty:
        return pd.DataFrame()

    summary = (
        influence_df.groupby(
            ["window_label", "window_time_range", "start_idx", "end_idx"],
            dropna=False,
        )
        .agg(
            mean_abs_delta=("abs_delta", "mean"),
            max_abs_delta=("abs_delta", "max"),
            mean_delta=("delta", "mean"),
        )
        .reset_index()
        .sort_values("mean_abs_delta", ascending=False)
    )
    return summary


def build_influence_heatmap(influence_df: pd.DataFrame, value_column: str) -> go.Figure:
    window_order = (
        influence_df[["window_label", "start_idx"]]
        .drop_duplicates()
        .sort_values("start_idx")["window_label"]
        .tolist()
    )
    horizon_order = (
        influence_df[["horizon_label", "horizon_index"]]
        .drop_duplicates()
        .sort_values("horizon_index")["horizon_label"]
        .tolist()
    )
    matrix = (
        influence_df.pivot_table(
            index="window_label",
            columns="horizon_label",
            values=value_column,
            aggfunc="mean",
        )
        .reindex(index=window_order, columns=horizon_order)
        .iloc[::-1]
    )

    heatmap_args = {
        "z": matrix.values,
        "x": matrix.columns.tolist(),
        "y": matrix.index.tolist(),
        "colorscale": "YlOrRd" if value_column == "abs_delta" else "RdBu_r",
        "colorbar": {"title": value_column},
        "hovertemplate": "window=%{y}<br>horizon=%{x}<br>value=%{z:.6g}<extra></extra>",
    }
    if value_column == "delta":
        heatmap_args["zmid"] = 0

    figure = go.Figure(data=go.Heatmap(**heatmap_args))
    figure.update_layout(
        height=max(360, min(760, 120 + 32 * len(matrix.index))),
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis_title="forecast horizon",
        yaxis_title="perturbed input window",
    )
    return figure


def build_influence_forecast_comparison(
    influence_df: pd.DataFrame,
    selected_window_label: str,
) -> go.Figure:
    selected_rows = influence_df.loc[influence_df["window_label"] == selected_window_label].copy()
    selected_rows["forecast_timestamp"] = pd.to_datetime(selected_rows["forecast_timestamp"], errors="coerce")
    selected_rows = selected_rows.sort_values("horizon_index")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=selected_rows["forecast_timestamp"],
            y=selected_rows["baseline_prediction"],
            mode="lines+markers",
            name="baseline",
            line={"color": "#264653", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected_rows["forecast_timestamp"],
            y=selected_rows["perturbed_prediction"],
            mode="lines+markers",
            name="perturbed",
            line={"color": "#e76f51", "width": 2},
        )
    )
    figure.update_layout(
        height=340,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h"},
    )
    return figure


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

        metrics = compute_metrics(
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


def build_forecast_weekday_comparison_plot(
    history_df: pd.DataFrame,
    comparison_pred_df: pd.DataFrame,
    actual_future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    target_column: str,
    selected_id: str,
    selected_scenarios: list[str],
) -> go.Figure:
    history = history_df.loc[
        history_df[id_column].astype(str) == selected_id,
        [timestamp_column, target_column],
    ].copy()
    history[timestamp_column] = pd.to_datetime(history[timestamp_column], errors="coerce")
    history = history.sort_values(timestamp_column)

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=history[timestamp_column],
            y=history[target_column],
            mode="lines",
            name="history",
            line={"color": "#264653", "width": 2},
        )
    )

    if actual_future_df is not None and not actual_future_df.empty:
        actual_future = actual_future_df.loc[
            actual_future_df[id_column].astype(str) == selected_id,
            [timestamp_column, target_column],
        ].copy()
        actual_future[timestamp_column] = pd.to_datetime(actual_future[timestamp_column], errors="coerce")
        actual_future = actual_future.sort_values(timestamp_column)
        figure.add_trace(
            go.Scattergl(
                x=actual_future[timestamp_column],
                y=actual_future[target_column],
                mode="lines",
                name="actual_future",
                line={"color": "#2a9d8f", "width": 2},
            )
        )

    for scenario in selected_scenarios:
        scenario_pred_df = comparison_pred_df.loc[
            (comparison_pred_df["scenario"].astype(str) == scenario)
            & (comparison_pred_df[id_column].astype(str) == selected_id)
        ].copy()
        if scenario_pred_df.empty:
            continue

        point_column = get_point_prediction_column(
            pred_df=scenario_pred_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )
        scenario_pred_df[timestamp_column] = pd.to_datetime(scenario_pred_df[timestamp_column], errors="coerce")
        scenario_pred_df = scenario_pred_df.sort_values(timestamp_column)
        figure.add_trace(
            go.Scattergl(
                x=scenario_pred_df[timestamp_column],
                y=scenario_pred_df[point_column],
                mode="lines",
                name=scenario,
                line={"color": get_weekday_scenario_color(scenario), "width": 2},
            )
        )

    figure.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h"},
    )
    return figure


def build_sliding_windows(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    context_length: int,
    prediction_length: int,
    stride: int,
    max_windows: int | None = None,
) -> list[dict[str, object]]:
    prepared = context_df.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    prepared = prepared.sort_values([id_column, timestamp_column])

    lengths = prepared.groupby(id_column).size()
    if lengths.empty:
        raise ValueError("슬라이딩 검증을 수행할 데이터가 없습니다.")
    if stride < 1:
        raise ValueError("윈도우 간격은 1 이상이어야 합니다.")
    if lengths.min() < context_length + prediction_length:
        min_length = int(lengths.min())
        required_length = int(context_length + prediction_length)
        shortest_id = str(lengths.idxmin())
        max_context_length = max(0, min_length - prediction_length)
        raise ValueError(
            "선택한 데이터 길이가 부족합니다. "
            f"현재 선택 구간의 최소 시계열 길이는 {min_length}개(`{shortest_id}`)이고, "
            f"현재 검증 설정은 최소 {required_length}개"
            f"({context_length} history + {prediction_length} forecast)가 필요합니다. "
            f"검증용 과거 길이를 {max_context_length} 이하로 줄이거나 분석 구간을 늘려주세요."
        )

    prepared_future = None
    if future_df is not None and not future_df.empty:
        prepared_future = future_df.copy()
        prepared_future[timestamp_column] = pd.to_datetime(prepared_future[timestamp_column], errors="coerce")

    positions = prepared.groupby(id_column).cumcount()
    common_length = int(lengths.min())
    max_start = common_length - context_length - prediction_length
    representative_id = str(prepared[id_column].iloc[0])
    windows: list[dict[str, object]] = []

    for window_idx, start_idx in enumerate(range(0, max_start + 1, stride), start=1):
        context_end = start_idx + context_length
        future_end = context_end + prediction_length
        model_context = prepared.loc[(positions >= start_idx) & (positions < context_end)].copy()
        actual_future = prepared.loc[(positions >= context_end) & (positions < future_end)].copy()

        model_future = None
        if prepared_future is not None and not prepared_future.empty:
            horizon_keys = actual_future[[id_column, timestamp_column]].drop_duplicates()
            model_future = prepared_future.merge(horizon_keys, on=[id_column, timestamp_column], how="inner")

        ref_context = model_context.loc[model_context[id_column].astype(str) == representative_id]
        ref_future = actual_future.loc[actual_future[id_column].astype(str) == representative_id]
        windows.append(
            {
                "window_index": window_idx,
                "start_idx": start_idx,
                "end_idx": context_end - 1,
                "forecast_start": str(ref_future[timestamp_column].min()),
                "forecast_end": str(ref_future[timestamp_column].max()),
                "history_df": model_context,
                "future_df": model_future,
                "actual_df": actual_future,
            }
        )
        if max_windows is not None and len(windows) >= max_windows:
            break

    return windows


def estimate_sliding_window_count(
    series_length: int,
    context_length: int,
    prediction_length: int,
    stride: int,
    max_windows: int | None = None,
) -> int:
    if series_length < 1 or stride < 1:
        return 0
    max_start = series_length - context_length - prediction_length
    if max_start < 0:
        return 0

    window_count = (max_start // stride) + 1
    if max_windows is not None:
        return min(window_count, max_windows)
    return window_count


def run_sliding_window_validation(
    pipeline,
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
    target_column: str,
    context_length: int,
    prediction_length: int,
    stride: int,
    max_windows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = build_sliding_windows(
        context_df=context_df,
        future_df=future_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        context_length=context_length,
        prediction_length=prediction_length,
        stride=stride,
        max_windows=max_windows,
    )

    window_rows: list[dict[str, object]] = []
    history_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    actual_frames: list[pd.DataFrame] = []
    for window in windows:
        history_df = trim_context_to_model_limit(
            context_df=window["history_df"],
            id_column=id_column,
            timestamp_column=timestamp_column,
        )
        pred_df = run_prediction(
            pipeline=pipeline,
            context_df=history_df,
            future_df=window["future_df"],
            prediction_length=prediction_length,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_column=target_column,
        )
        metrics = compute_metrics(
            pred_df=pred_df,
            history_df=history_df,
            actual_df=window["actual_df"],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_column=target_column,
        ).iloc[0].to_dict()
        history_with_window = history_df.copy()
        history_with_window["window_index"] = int(window["window_index"])
        pred_with_window = pred_df.copy()
        pred_with_window["window_index"] = int(window["window_index"])
        actual_with_window = window["actual_df"].copy()
        actual_with_window["window_index"] = int(window["window_index"])
        history_frames.append(history_with_window)
        pred_frames.append(pred_with_window)
        actual_frames.append(actual_with_window)
        window_rows.append(
            {
                "window_index": int(window["window_index"]),
                "start_idx": int(window["start_idx"]),
                "end_idx": int(window["end_idx"]),
                "forecast_start": window["forecast_start"],
                "forecast_end": window["forecast_end"],
                **metrics,
            }
        )

    windows_df = pd.DataFrame(window_rows)
    summary_df = pd.DataFrame(
        {
            "windows": [int(len(windows_df))],
            "MAE_mean": [float(windows_df["MAE"].mean())],
            "MAE_std": [float(windows_df["MAE"].std(ddof=0))],
            "RMSE_mean": [float(windows_df["RMSE"].mean())],
            "RMSE_std": [float(windows_df["RMSE"].std(ddof=0))],
            "MAPE_mean": [float(windows_df["MAPE"].mean())],
            "MAPE_std": [float(windows_df["MAPE"].std(ddof=0))],
            "MASE_mean": [float(windows_df["MASE"].mean())],
            "MASE_std": [float(windows_df["MASE"].std(ddof=0))],
            "Correlation_mean": [float(windows_df["Correlation"].mean())],
            "Correlation_std": [float(windows_df["Correlation"].std(ddof=0))],
        }
    )
    history_detail_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    pred_detail_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    actual_detail_df = pd.concat(actual_frames, ignore_index=True) if actual_frames else pd.DataFrame()
    return summary_df, windows_df, history_detail_df, pred_detail_df, actual_detail_df


def build_validation_comparison_summary(validation_windows_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["MAE", "RMSE", "MAPE", "MASE", "Correlation"]
    if validation_windows_df.empty or "scenario" not in validation_windows_df.columns:
        return pd.DataFrame()

    grouped = validation_windows_df.groupby("scenario", dropna=False)[metric_columns].mean().reset_index()
    if WEEKDAY_BASELINE_SCENARIO not in set(grouped["scenario"]):
        return grouped

    baseline = grouped.loc[grouped["scenario"] == WEEKDAY_BASELINE_SCENARIO].iloc[0]
    comparison_rows: list[dict[str, object]] = []

    for _, scenario_row in grouped.iterrows():
        scenario_name = str(scenario_row["scenario"])
        if scenario_name == WEEKDAY_BASELINE_SCENARIO:
            continue
        for metric in metric_columns:
            delta = float(scenario_row[metric] - baseline[metric])
            if metric == "Correlation":
                better = "개선" if delta > 0 else "악화" if delta < 0 else "동일"
            else:
                better = "개선" if delta < 0 else "악화" if delta > 0 else "동일"
            comparison_rows.append(
                {
                    "scenario": scenario_name,
                    "metric": metric,
                    WEEKDAY_BASELINE_SCENARIO: float(baseline[metric]),
                    "비교 시나리오": float(scenario_row[metric]),
                    "delta": delta,
                    "판정": better,
                }
            )

    return pd.DataFrame(comparison_rows)


def show_top_status(
    context_df: pd.DataFrame,
    prediction_length: int,
    device: str,
    id_column: str,
    timestamp_column: str,
) -> None:
    series_count = 0
    if not context_df.empty and id_column and timestamp_column:
        series_count = int(get_series_lengths(context_df, id_column, timestamp_column).shape[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("시계열 수", series_count)
    col2.metric("전역 예측 구간 길이", int(prediction_length))
    col3.metric("실행 장치", device)


def show_metric_cards(metrics_df: pd.DataFrame) -> None:
    if metrics_df.empty:
        return
    metric_names = ["MAE", "RMSE", "MAPE", "MASE", "Correlation"]
    columns = st.columns(len(metric_names))
    row = metrics_df.iloc[0]
    for column, metric_name in zip(columns, metric_names):
        value = row.get(metric_name)
        display = "-" if pd.isna(value) else f"{float(value):.4f}"
        column.metric(metric_name, display)


def show_validation_cards(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    row = summary_df.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("윈도우 수", int(row["windows"]))
    col2.metric("평균 MAE", f"{float(row['MAE_mean']):.4f}")
    col3.metric("평균 RMSE", f"{float(row['RMSE_mean']):.4f}")
    col4.metric("평균 MASE", f"{float(row['MASE_mean']):.4f}")


with st.sidebar:
    st.header("빠른 안내")
    st.caption("Forecast, Validation, Covariate Lab 탭에서 실험을 이어갈 수 있습니다.")
    st.markdown(
        "- 과거 데이터는 필수입니다.\n"
        "- 미래 공변량은 선택입니다.\n"
        "- `id`가 없으면 단일 시계열로 자동 생성할 수 있습니다.\n"
        f"- Chronos-2 입력 history는 시계열당 최대 {CHRONOS2_MAX_CONTEXT_LENGTH}개입니다."
    )
    st.markdown("**입력 형식 힌트**")
    st.code("과거 데이터: id, timestamp, target, covariates...", language="text")
    st.code("미래 공변량: id, timestamp, known_covariates...", language="text")

device_options = []
for candidate in [detect_device(), "cpu", "mps", "cuda"]:
    if candidate not in device_options:
        device_options.append(candidate)

context_df = pd.DataFrame()
full_context_df = pd.DataFrame()
future_df = None
id_column = ""
timestamp_column = ""
target_column = ""
experiment_mode = "선택 구간 끝에서 미래 예측"

default_prediction_length = int(st.session_state.get("prediction_length_input", 24))
default_model_id = str(st.session_state.get("model_id_input", "amazon/chronos-2"))
default_device = str(st.session_state.get("device_input", device_options[0]))

st.markdown("로컬에서 예측, 검증, 공변량 실험을 빠르게 반복하는 시계열 워크벤치")

global_settings = st.container(border=True)
with global_settings:
    st.markdown("**전역 설정**")
    setting_col1, setting_col2, setting_col3 = st.columns([2, 1, 1])
    with setting_col1:
        model_id = st.text_input("모델 ID", value=default_model_id, key="model_id_input")
    with setting_col2:
        device_index = device_options.index(default_device) if default_device in device_options else 0
        device = st.selectbox("실행 장치", options=device_options, index=device_index, key="device_input")
    with setting_col3:
        prediction_length = st.number_input(
            "예측 구간 길이",
            min_value=1,
            max_value=CHRONOS2_MAX_PREDICTION_LENGTH,
            value=default_prediction_length,
            step=1,
            key="prediction_length_input",
        )

tabs = st.tabs(["Forecast", "Validation", "Covariate Lab"])

with tabs[0]:
    st.subheader("Forecast")
    st.caption("단일 예측을 실행하고 결과를 빠르게 확인합니다.")

    prep_col, inspect_col = st.columns([1.05, 1.15])
    with prep_col:
        with st.container(border=True):
            st.markdown("**1. 데이터 준비**")
            st.caption("과거 데이터는 필수이고, 미래 공변량은 선택입니다. CSV, Parquet, TSF를 지원합니다.")
            context_file = st.file_uploader("과거 데이터", type=["csv", "parquet", "pq", "tsf"])
            future_file = st.file_uploader("미래 공변량", type=["csv", "parquet", "pq", "tsf"])

            if context_file:
                context_df = load_uploaded_table(context_file.name, context_file.getvalue())
                full_context_df = context_df.copy()
                if future_file:
                    future_df = load_uploaded_table(future_file.name, future_file.getvalue())

    with inspect_col:
        with st.container(border=True):
            st.markdown("**2. 입력 확인**")
            if not context_df.empty:
                st.markdown("스키마 매핑")
                guessed_id_column = guess_id_column(context_df)
                guessed_timestamp_column = guess_timestamp_column(context_df)
                guessed_target_column = guess_target_column(context_df)

                id_options = [AUTO_ID_OPTION, *context_df.columns.tolist()]
                id_index = 0 if guessed_id_column == AUTO_ID_OPTION else context_df.columns.get_loc(guessed_id_column) + 1

                meta1, meta2, meta3 = st.columns(3)
                with meta1:
                    id_column = st.selectbox(
                        "시계열 ID 컬럼",
                        options=id_options,
                        index=id_index,
                        format_func=lambda value: "단일 시계열로 자동 생성" if value == AUTO_ID_OPTION else value,
                    )
                with meta2:
                    timestamp_column = st.selectbox(
                        "시간 컬럼",
                        options=context_df.columns,
                        index=context_df.columns.get_loc(guessed_timestamp_column),
                    )
                with meta3:
                    target_column = st.selectbox(
                        "타깃 컬럼",
                        options=context_df.columns,
                        index=context_df.columns.get_loc(guessed_target_column),
                    )

                context_df, future_df, id_column = apply_id_selection(
                    context_df=context_df,
                    future_df=future_df,
                    id_selection=id_column,
                )
                full_context_df = context_df.copy()

                available_context_covariates = [
                    col for col in context_df.columns if col not in {id_column, timestamp_column, target_column}
                ]
                available_known_future_covariates: list[str] = []
                if future_df is not None and not future_df.empty:
                    available_known_future_covariates = [
                        col
                        for col in future_df.columns
                        if col in available_context_covariates and col not in {id_column, timestamp_column, target_column}
                    ]

                selected_known_future_covariates = st.multiselect(
                    "미래 known 공변량",
                    options=available_known_future_covariates,
                    default=[],
                    help="예측 시점에도 미리 알고 있는 값만 선택합니다. 선택한 컬럼은 context와 future에 함께 사용됩니다.",
                )
                available_past_only_covariates = [
                    col for col in available_context_covariates if col not in set(selected_known_future_covariates)
                ]
                selected_past_only_covariates = st.multiselect(
                    "과거 전용 공변량",
                    options=available_past_only_covariates,
                    default=[],
                    help="과거 구간에서만 관측 가능한 값입니다. future에는 전달되지 않습니다.",
                )
                context_df, full_context_df, future_df = filter_model_columns(
                    context_df=context_df,
                    full_context_df=full_context_df,
                    future_df=future_df,
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                    target_column=target_column,
                    past_only_covariates=selected_past_only_covariates,
                    known_future_covariates=selected_known_future_covariates,
                )

                st.markdown("분석 구간 선택")
                lengths = get_series_lengths(context_df, id_column=id_column, timestamp_column=timestamp_column)
                inspect_ids = context_df[id_column].astype(str).drop_duplicates().tolist()
                inspect_id = st.selectbox("확인용 시계열", options=inspect_ids)
                inspect_timestamps = get_index_timestamp_map(
                    df=context_df,
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                    selected_id=inspect_id,
                )

                info1, info2, info3 = st.columns(3)
                info1.metric("전체 시작", str(inspect_timestamps.iloc[0]))
                info2.metric("전체 종료", str(inspect_timestamps.iloc[-1]))
                info3.metric("전체 길이", int(len(inspect_timestamps)))

                shared_length = int(lengths.min())
                range_col1, range_col2 = st.columns(2)
                with range_col1:
                    start_idx = st.number_input("시작 인덱스", min_value=0, max_value=shared_length - 1, value=0, step=1)
                with range_col2:
                    end_idx = st.number_input(
                        "종료 인덱스",
                        min_value=int(start_idx),
                        max_value=shared_length - 1,
                        value=shared_length - 1,
                        step=1,
                    )

                ts_col1, ts_col2 = st.columns(2)
                ts_col1.info(f"시작 시점: {inspect_timestamps.iloc[int(start_idx)]}")
                ts_col2.info(f"종료 시점: {inspect_timestamps.iloc[int(end_idx)]}")

                experiment_mode = st.radio(
                    "실험 모드",
                    options=["데이터셋 내부 평가", "선택 구간 끝에서 미래 예측"],
                    index=0,
                )

                context_df = trim_by_index_window(
                    df=context_df,
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                )
            else:
                st.info("과거 데이터를 업로드하면 스키마 매핑과 분석 구간 선택이 열립니다.")

    can_run = not context_df.empty
    if can_run:
        show_top_status(
            context_df=context_df,
            prediction_length=int(prediction_length),
            device=device,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )

        if int(get_series_lengths(context_df, id_column=id_column, timestamp_column=timestamp_column).max()) > CHRONOS2_MAX_CONTEXT_LENGTH:
            st.warning(
                "Chronos-2 공식 스펙에 맞추기 위해 각 시계열의 최근 "
                f"{CHRONOS2_MAX_CONTEXT_LENGTH}개 시점만 모델 입력에 사용합니다."
            )

        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            with st.container(border=True):
                st.markdown("**입력 컨텍스트 미리보기**")
                show_context_summary(
                    context_df=context_df,
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                )
                st.dataframe(context_df.head(10), use_container_width=True)
        with preview_col2:
            with st.container(border=True):
                st.markdown("**미래 공변량 미리보기**")
                if future_df is None or future_df.empty:
                    st.info("미래 공변량 없이 실행합니다.")
                else:
                    st.dataframe(future_df.head(10), use_container_width=True)

    with st.container(border=True):
        st.markdown("**3. 실행 설정**")
        exec_col1, exec_col2, exec_col3 = st.columns(3)
        exec_col1.metric("모델 ID", model_id)
        exec_col2.metric("실행 장치", device)
        exec_col3.metric("예측 구간 길이", int(prediction_length))
        compare_forecast_weekday_covariate = st.checkbox(
            "요일 공변량 Forecast 비교",
            value=False,
            help=(
                "기본 예측과 함께 문자열, 0~6, sin/cos 요일 인코딩을 "
                "`과거만`/`과거+미래` 범위로 모두 비교합니다."
            ),
        )
        if compare_forecast_weekday_covariate:
            st.caption("baseline 1회 + 요일 인코딩 3종 x 적용 범위 2종으로 최대 7개 forecast를 실행합니다.")

        if st.button("Chronos-2 예측 실행", type="primary", disabled=not can_run):
            try:
                with st.spinner("모델을 불러오고 예측하는 중입니다..."):
                    pipeline = get_pipeline(model_id, device)
                    model_context_df, model_future_df, actual_future_df = prepare_model_inputs_for_forecast(
                        context_df=context_df,
                        full_context_df=full_context_df,
                        future_df=future_df,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        prediction_length=int(prediction_length),
                        experiment_mode=experiment_mode,
                    )

                    st.session_state["pred_df"] = run_prediction(
                        pipeline=pipeline,
                        context_df=model_context_df,
                        future_df=model_future_df,
                        prediction_length=int(prediction_length),
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        target_column=target_column,
                    )
                    st.session_state["result_meta"] = {
                        "id_column": id_column,
                        "timestamp_column": timestamp_column,
                        "target_column": target_column,
                        "history_df": model_context_df,
                        "model_future_df": model_future_df,
                        "actual_future_df": actual_future_df,
                    }
                    if actual_future_df is not None and not actual_future_df.empty:
                        st.session_state["metrics_df"] = compute_metrics(
                            pred_df=st.session_state["pred_df"],
                            history_df=model_context_df,
                            actual_df=actual_future_df,
                            id_column=id_column,
                            timestamp_column=timestamp_column,
                            target_column=target_column,
                        )
                    else:
                        st.session_state["metrics_df"] = None

                    if compare_forecast_weekday_covariate:
                        (
                            weekday_comparison_pred_df,
                            weekday_comparison_metrics_df,
                            weekday_comparison_errors_df,
                        ) = run_forecast_weekday_comparison(
                            pipeline=pipeline,
                            baseline_pred_df=st.session_state["pred_df"],
                            model_context_df=model_context_df,
                            model_future_df=model_future_df,
                            actual_future_df=actual_future_df,
                            prediction_length=int(prediction_length),
                            id_column=id_column,
                            timestamp_column=timestamp_column,
                            target_column=target_column,
                        )
                        st.session_state["forecast_weekday_comparison_pred_df"] = weekday_comparison_pred_df
                        st.session_state["forecast_weekday_comparison_metrics_df"] = weekday_comparison_metrics_df
                        st.session_state["forecast_weekday_comparison_errors_df"] = weekday_comparison_errors_df
                    else:
                        st.session_state["forecast_weekday_comparison_pred_df"] = None
                        st.session_state["forecast_weekday_comparison_metrics_df"] = None
                        st.session_state["forecast_weekday_comparison_errors_df"] = None

                st.success("예측이 완료되었습니다.")
            except Exception as exc:
                st.error(f"실행 중 오류가 발생했습니다: {exc}")

    pred_df = st.session_state.get("pred_df")
    result_meta = st.session_state.get("result_meta")
    metrics_df = st.session_state.get("metrics_df")
    if isinstance(pred_df, pd.DataFrame) and not pred_df.empty and result_meta:
        with st.container(border=True):
            st.markdown("**4. 결과 요약**")
            if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
                show_metric_cards(metrics_df)
            else:
                st.info("실측 미래 구간이 없어서 평가 지표는 계산하지 않았습니다.")

        with st.container(border=True):
            st.markdown("**대표 시계열 미리보기**")
            show_series_preview(
                context_df=result_meta["history_df"],
                pred_df=pred_df,
                actual_future_df=result_meta["actual_future_df"],
                id_column=result_meta["id_column"],
                timestamp_column=result_meta["timestamp_column"],
                target_column=result_meta["target_column"],
            )

        result_col1, result_col2 = st.columns([1.25, 1.0])
        with result_col1:
            with st.container(border=True):
                st.markdown("**예측 결과 테이블**")
                st.dataframe(pred_df.head(50), use_container_width=True)
        with result_col2:
            with st.container(border=True):
                show_model_input_summary(
                    context_df=result_meta["history_df"],
                    future_df=result_meta["model_future_df"],
                    id_column=result_meta["id_column"],
                    timestamp_column=result_meta["timestamp_column"],
                    target_column=result_meta["target_column"],
                )
                save_predictions_download(pred_df)

        weekday_comparison_pred_df = st.session_state.get("forecast_weekday_comparison_pred_df")
        weekday_comparison_metrics_df = st.session_state.get("forecast_weekday_comparison_metrics_df")
        weekday_comparison_errors_df = st.session_state.get("forecast_weekday_comparison_errors_df")
        if isinstance(weekday_comparison_pred_df, pd.DataFrame) and not weekday_comparison_pred_df.empty:
            with st.container(border=True):
                st.markdown("**요일 공변량 Forecast 비교**")
                if (
                    isinstance(weekday_comparison_metrics_df, pd.DataFrame)
                    and not weekday_comparison_metrics_df.empty
                ):
                    st.dataframe(
                        build_forecast_weekday_metric_summary(weekday_comparison_metrics_df),
                        use_container_width=True,
                    )
                else:
                    st.info("실측 미래 구간이 없어서 시나리오별 평가 지표는 계산하지 않았습니다.")

                if (
                    isinstance(weekday_comparison_errors_df, pd.DataFrame)
                    and not weekday_comparison_errors_df.empty
                ):
                    st.warning("일부 요일 공변량 시나리오는 실행하지 못했습니다.")
                    st.dataframe(weekday_comparison_errors_df, use_container_width=True)

                comparison_col1, comparison_col2 = st.columns([0.35, 0.65])
                comparison_ids = result_meta["history_df"][result_meta["id_column"]].astype(str).unique().tolist()
                selected_comparison_id = comparison_col1.selectbox(
                    "비교 시계열",
                    options=comparison_ids,
                    key="forecast_weekday_comparison_id",
                )
                scenario_options = weekday_comparison_pred_df["scenario"].dropna().astype(str).unique().tolist()
                selected_comparison_scenarios = comparison_col2.multiselect(
                    "비교 시나리오",
                    options=scenario_options,
                    default=scenario_options,
                    key="forecast_weekday_comparison_scenarios",
                )
                if selected_comparison_scenarios:
                    comparison_figure = build_forecast_weekday_comparison_plot(
                        history_df=result_meta["history_df"],
                        comparison_pred_df=weekday_comparison_pred_df,
                        actual_future_df=result_meta["actual_future_df"],
                        id_column=result_meta["id_column"],
                        timestamp_column=result_meta["timestamp_column"],
                        target_column=result_meta["target_column"],
                        selected_id=selected_comparison_id,
                        selected_scenarios=selected_comparison_scenarios,
                    )
                    st.plotly_chart(comparison_figure, use_container_width=True)

                with st.expander("요일 공변량 비교 예측 데이터"):
                    st.dataframe(weekday_comparison_pred_df, use_container_width=True)

with tabs[1]:
    st.subheader("Validation")
    st.caption("현재 준비된 데이터 구간으로 슬라이딩 윈도우 검증을 수행합니다.")

    show_top_status(
        context_df=context_df,
        prediction_length=int(prediction_length),
        device=device,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    validation_lengths = pd.Series(dtype="int64")
    validation_min_length = 0
    if can_run and id_column and timestamp_column:
        validation_lengths = get_series_lengths(
            context_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )
        if not validation_lengths.empty:
            validation_min_length = int(validation_lengths.min())

    val_cfg_col1, val_cfg_col2 = st.columns(2)
    with val_cfg_col1:
        with st.container(border=True):
            st.markdown("**1. 검증 설정**")
            validation_min_context_length = 8
            validation_prediction_max = CHRONOS2_MAX_PREDICTION_LENGTH
            if validation_min_length > 0:
                validation_prediction_max = min(
                    CHRONOS2_MAX_PREDICTION_LENGTH,
                    max(1, validation_min_length - validation_min_context_length),
                )
            validation_prediction_default = min(int(prediction_length), int(validation_prediction_max))
            validation_prediction_length = st.number_input(
                "검증용 예측 구간 길이",
                min_value=1,
                max_value=int(validation_prediction_max),
                value=int(validation_prediction_default),
                step=1,
            )

            validation_context_max = CHRONOS2_MAX_CONTEXT_LENGTH
            if validation_min_length > 0:
                validation_context_max = min(
                    CHRONOS2_MAX_CONTEXT_LENGTH,
                    max(
                        validation_min_context_length,
                        validation_min_length - int(validation_prediction_length),
                    ),
                )
            validation_context_default = min(168, int(validation_context_max))
            validation_context_length = st.number_input(
                "검증용 과거 길이",
                min_value=validation_min_context_length,
                max_value=int(validation_context_max),
                value=int(validation_context_default),
                step=8,
            )
            validation_stride = st.number_input(
                "윈도우 간격",
                min_value=1,
                max_value=CHRONOS2_MAX_CONTEXT_LENGTH,
                value=1,
                step=1,
            )
            max_windows_input = st.number_input(
                "최대 윈도우 수 (0이면 전체)",
                min_value=0,
                max_value=10000,
                value=0,
                step=1,
            )
            max_windows = None if int(max_windows_input) == 0 else int(max_windows_input)

            validation_required_length = int(validation_context_length) + int(validation_prediction_length)
            validation_ready = bool(can_run and validation_min_length >= validation_required_length)
            estimated_validation_windows = estimate_sliding_window_count(
                series_length=validation_min_length,
                context_length=int(validation_context_length),
                prediction_length=int(validation_prediction_length),
                stride=int(validation_stride),
                max_windows=max_windows,
            )
            status_col1, status_col2, status_col3 = st.columns(3)
            status_col1.metric("선택 구간 최소 길이", int(validation_min_length))
            status_col2.metric("필요 길이", int(validation_required_length))
            status_col3.metric("예상 윈도우 수", int(estimated_validation_windows))
            if can_run and not validation_ready:
                st.warning(
                    "현재 검증 설정을 실행하려면 선택 구간을 더 길게 잡거나 "
                    "검증용 과거 길이 또는 검증용 예측 구간 길이를 줄여주세요."
                )
    with val_cfg_col2:
        with st.container(border=True):
            st.markdown("**2. 비교 옵션**")
            compare_weekday_covariate = st.checkbox(
                "요일 공변량 A/B 비교",
                value=False,
                help=(
                    f"timestamp에서 요일을 자동 생성해 `{WEEKDAY_BASELINE_SCENARIO}`와 "
                    "`曜日-有(인코딩,過去/過去+未来)`를 비교합니다."
                ),
            )
            if compare_weekday_covariate:
                selected_weekday_encoding = st.selectbox(
                    "요일 인코딩",
                    options=["문자열", "0~6", "sin/cos"],
                    index=0,
                    help="한 번에 한 가지 인코딩을 고르고, 적용 범위만 3가지 시나리오로 비교합니다.",
                )
                st.caption(
                    f"기준선은 `{WEEKDAY_BASELINE_SCENARIO}`이고, 같은 인코딩으로 "
                    "`過去`와 `過去+未来`를 함께 비교합니다."
                )
            else:
                st.caption(
                    "선택하면 기본 검증 외에도 요일 정보를 covariate로 추가한 시나리오를 함께 계산합니다."
                )

    validation_can_run = bool(can_run and validation_ready)

    if st.button("슬라이딩 윈도우 검증 실행", type="primary", disabled=not validation_can_run):
        try:
            with st.spinner("슬라이딩 윈도우 검증을 수행하는 중입니다..."):
                pipeline = get_pipeline(model_id, device)
                summary_df, windows_df, validation_history_df, validation_pred_df, validation_actual_df = run_sliding_window_validation(
                    pipeline=pipeline,
                    context_df=context_df,
                    future_df=future_df,
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                    target_column=target_column,
                    context_length=int(validation_context_length),
                    prediction_length=int(validation_prediction_length),
                    stride=int(validation_stride),
                    max_windows=max_windows,
                )
                summary_df = summary_df.copy()
                windows_df = windows_df.copy()
                validation_history_df = validation_history_df.copy()
                validation_pred_df = validation_pred_df.copy()
                validation_actual_df = validation_actual_df.copy()
                summary_df["scenario"] = WEEKDAY_BASELINE_SCENARIO
                windows_df["scenario"] = WEEKDAY_BASELINE_SCENARIO
                validation_history_df["scenario"] = WEEKDAY_BASELINE_SCENARIO
                validation_pred_df["scenario"] = WEEKDAY_BASELINE_SCENARIO
                validation_actual_df["scenario"] = WEEKDAY_BASELINE_SCENARIO

                comparison_summary_df = pd.DataFrame()
                if compare_weekday_covariate:
                    for _, include_future_covariate in WEEKDAY_COVARIATE_SCOPES:
                        scenario_name = build_weekday_scenario_name(
                            selected_weekday_encoding,
                            include_future_covariate,
                        )
                        weekday_context_df, weekday_future_df = build_weekday_covariate_frames(
                            context_df=context_df,
                            future_df=future_df,
                            id_column=id_column,
                            timestamp_column=timestamp_column,
                            encoding=selected_weekday_encoding,
                            include_future_covariate=include_future_covariate,
                        )
                        weekday_summary_df, weekday_windows_df, weekday_history_df, weekday_pred_df, weekday_actual_df = run_sliding_window_validation(
                            pipeline=pipeline,
                            context_df=weekday_context_df,
                            future_df=weekday_future_df,
                            id_column=id_column,
                            timestamp_column=timestamp_column,
                            target_column=target_column,
                            context_length=int(validation_context_length),
                            prediction_length=int(validation_prediction_length),
                            stride=int(validation_stride),
                            max_windows=max_windows,
                        )
                        weekday_summary_df = weekday_summary_df.copy()
                        weekday_windows_df = weekday_windows_df.copy()
                        weekday_history_df = weekday_history_df.copy()
                        weekday_pred_df = weekday_pred_df.copy()
                        weekday_actual_df = weekday_actual_df.copy()
                        weekday_summary_df["scenario"] = scenario_name
                        weekday_windows_df["scenario"] = scenario_name
                        weekday_history_df["scenario"] = scenario_name
                        weekday_pred_df["scenario"] = scenario_name
                        weekday_actual_df["scenario"] = scenario_name
                        summary_df = pd.concat([summary_df, weekday_summary_df], ignore_index=True)
                        windows_df = pd.concat([windows_df, weekday_windows_df], ignore_index=True)
                        validation_history_df = pd.concat([validation_history_df, weekday_history_df], ignore_index=True)
                        validation_pred_df = pd.concat([validation_pred_df, weekday_pred_df], ignore_index=True)
                        validation_actual_df = pd.concat([validation_actual_df, weekday_actual_df], ignore_index=True)
                    comparison_summary_df = build_validation_comparison_summary(windows_df)

                st.session_state["validation_summary_df"] = summary_df
                st.session_state["validation_windows_df"] = windows_df
                st.session_state["validation_comparison_summary_df"] = comparison_summary_df
                st.session_state["validation_history_detail_df"] = validation_history_df
                st.session_state["validation_pred_detail_df"] = validation_pred_df
                st.session_state["validation_actual_detail_df"] = validation_actual_df
            st.success("자동 검증이 완료되었습니다.")
        except Exception as exc:
            st.error(f"자동 검증 중 오류가 발생했습니다: {exc}")

    validation_summary_df = st.session_state.get("validation_summary_df")
    validation_windows_df = st.session_state.get("validation_windows_df")
    validation_comparison_summary_df = st.session_state.get("validation_comparison_summary_df")
    validation_history_detail_df = st.session_state.get("validation_history_detail_df")
    validation_pred_detail_df = st.session_state.get("validation_pred_detail_df")
    validation_actual_detail_df = st.session_state.get("validation_actual_detail_df")

    if isinstance(validation_summary_df, pd.DataFrame) and not validation_summary_df.empty:
        with st.container(border=True):
            st.markdown("**검증 요약**")
            base_summary_df = validation_summary_df.loc[
                validation_summary_df["scenario"] == WEEKDAY_BASELINE_SCENARIO
            ]
            if not base_summary_df.empty:
                show_validation_cards(base_summary_df)
            st.dataframe(validation_summary_df, use_container_width=True)

    if isinstance(validation_windows_df, pd.DataFrame) and not validation_windows_df.empty:
        with st.container(border=True):
            st.markdown("**지표 추이**")
            metric_for_plot = st.selectbox(
                "추이로 볼 지표",
                options=["MAE", "RMSE", "MAPE", "MASE", "Correlation"],
                index=0,
            )
            trend_fig = go.Figure()
            if "scenario" in validation_windows_df.columns:
                for scenario_name, group in validation_windows_df.groupby("scenario", dropna=False):
                    group = group.sort_values("window_index")
                    trend_fig.add_trace(
                        go.Scatter(
                            x=group["window_index"],
                            y=group[metric_for_plot],
                            mode="lines+markers",
                            name=str(scenario_name),
                            line={"color": get_weekday_scenario_color(str(scenario_name)), "width": 2},
                        )
                    )
            else:
                trend_fig.add_trace(
                    go.Scatter(
                        x=validation_windows_df["window_index"],
                        y=validation_windows_df[metric_for_plot],
                        mode="lines+markers",
                        name=metric_for_plot,
                        line={"color": "#1d3557", "width": 2},
                    )
                )
            trend_fig.update_layout(
                height=360,
                margin={"l": 20, "r": 20, "t": 30, "b": 20},
                xaxis_title="window_index",
                yaxis_title=metric_for_plot,
            )
            st.plotly_chart(trend_fig, use_container_width=True)

    detail_col1, detail_col2 = st.columns([0.9, 1.2])
    with detail_col1:
        if isinstance(validation_comparison_summary_df, pd.DataFrame) and not validation_comparison_summary_df.empty:
            with st.container(border=True):
                st.markdown("**요일 공변량 비교**")
                st.dataframe(validation_comparison_summary_df, use_container_width=True)
    with detail_col2:
        if isinstance(validation_windows_df, pd.DataFrame) and not validation_windows_df.empty:
            with st.container(border=True):
                st.markdown("**윈도우별 검증 결과**")
                st.dataframe(validation_windows_df, use_container_width=True)

    if (
        isinstance(validation_history_detail_df, pd.DataFrame)
        and not validation_history_detail_df.empty
        and isinstance(validation_pred_detail_df, pd.DataFrame)
        and not validation_pred_detail_df.empty
        and isinstance(validation_actual_detail_df, pd.DataFrame)
        and not validation_actual_detail_df.empty
    ):
        with st.container(border=True):
            st.markdown("**시나리오별 Actual vs Prediction**")
            preview_col1, preview_col2, preview_col3 = st.columns(3)
            scenario_options = validation_pred_detail_df["scenario"].dropna().astype(str).unique().tolist()
            selected_validation_scenario = preview_col1.selectbox(
                "시나리오",
                options=scenario_options,
                key="validation_preview_scenario",
            )
            scenario_pred_df = validation_pred_detail_df.loc[
                validation_pred_detail_df["scenario"].astype(str) == selected_validation_scenario
            ].copy()
            scenario_history_df = validation_history_detail_df.loc[
                validation_history_detail_df["scenario"].astype(str) == selected_validation_scenario
            ].copy()
            scenario_actual_df = validation_actual_detail_df.loc[
                validation_actual_detail_df["scenario"].astype(str) == selected_validation_scenario
            ].copy()

            window_options = sorted(scenario_pred_df["window_index"].dropna().astype(int).unique().tolist())
            selected_validation_window = preview_col2.selectbox(
                "윈도우",
                options=window_options,
                key="validation_preview_window",
            )
            selected_history_df = scenario_history_df.loc[
                scenario_history_df["window_index"].astype(int) == selected_validation_window
            ].copy()
            selected_pred_df = scenario_pred_df.loc[
                scenario_pred_df["window_index"].astype(int) == selected_validation_window
            ].copy()
            selected_actual_df = scenario_actual_df.loc[
                scenario_actual_df["window_index"].astype(int) == selected_validation_window
            ].copy()

            available_ids = selected_history_df[id_column].astype(str).unique().tolist()
            selected_validation_id = preview_col3.selectbox(
                "시계열",
                options=available_ids,
                key="validation_preview_series",
            )
            preview_figure = build_plot_frame(
                history_df=selected_history_df,
                pred_df=selected_pred_df,
                actual_future_df=selected_actual_df,
                id_column=id_column,
                timestamp_column=timestamp_column,
                target_column=target_column,
                selected_id=selected_validation_id,
            )
            st.plotly_chart(preview_figure, use_container_width=True)

with tabs[2]:
    st.subheader("Covariate Lab")
    st.caption("선택한 입력 구간을 perturb해서 예측 민감도와 attention-like influence를 확인합니다.")

    influence_context_df = pd.DataFrame()
    influence_future_df = None
    influence_actual_future_df = None
    influence_inputs_ready = False
    influence_input_error = ""

    if can_run:
        try:
            influence_context_df, influence_future_df, influence_actual_future_df = prepare_model_inputs_for_forecast(
                context_df=context_df,
                full_context_df=full_context_df,
                future_df=future_df,
                id_column=id_column,
                timestamp_column=timestamp_column,
                prediction_length=int(prediction_length),
                experiment_mode=experiment_mode,
            )
            influence_inputs_ready = True
        except Exception as exc:
            influence_input_error = str(exc)

        with st.container(border=True):
            st.markdown("**현재 입력 요약**")
            if influence_inputs_ready:
                show_model_input_summary(
                    context_df=influence_context_df,
                    future_df=influence_future_df,
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                    target_column=target_column,
                )
            else:
                st.warning(f"현재 설정으로 모델 입력을 만들 수 없습니다: {influence_input_error}")
    else:
        st.info("과거 데이터를 업로드하면 Covariate Lab을 사용할 수 있습니다.")

    with st.container(border=True):
        st.markdown("**Attention-like Influence Map**")

        if not can_run:
            st.info("Forecast 탭에서 과거 데이터를 먼저 준비해주세요.")
        elif not influence_inputs_ready:
            st.warning(f"현재 설정으로 influence map을 계산할 수 없습니다: {influence_input_error}")
        else:
            influence_ids = influence_context_df[id_column].astype(str).drop_duplicates().tolist()
            input_source_options = [INFLUENCE_SOURCE_CONTEXT]
            if influence_future_df is not None and not influence_future_df.empty:
                input_source_options.append(INFLUENCE_SOURCE_FUTURE)

            cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
            selected_influence_id = cfg_col1.selectbox(
                "분석 시계열",
                options=influence_ids,
                key="influence_selected_id",
            )
            selected_input_source = cfg_col2.selectbox(
                "입력 소스",
                options=input_source_options,
                key="influence_input_source",
            )

            if selected_input_source == INFLUENCE_SOURCE_CONTEXT:
                source_df = influence_context_df
                context_feature_columns = [
                    col for col in influence_context_df.columns if col not in {id_column, timestamp_column}
                ]
                source_columns = []
                if target_column in context_feature_columns:
                    source_columns.append(target_column)
                source_columns.extend([col for col in context_feature_columns if col != target_column])
                source_key = "context"
            else:
                source_df = influence_future_df if influence_future_df is not None else pd.DataFrame()
                source_columns = [
                    col for col in source_df.columns if col not in {id_column, timestamp_column, target_column}
                ]
                source_key = "future"

            if not source_columns:
                st.info("선택한 입력 소스에는 perturb할 수 있는 컬럼이 없습니다.")
            else:
                perturb_column = cfg_col3.selectbox(
                    "Perturb 컬럼",
                    options=source_columns,
                    key=f"influence_perturb_column_{source_key}",
                )
                try:
                    source_series = get_series_rows_for_influence(
                        df=source_df,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        selected_id=selected_influence_id,
                    )
                    source_length = int(len(source_series))
                except Exception as exc:
                    source_series = pd.DataFrame()
                    source_length = 0
                    st.warning(f"선택한 입력 소스에서 시계열을 찾지 못했습니다: {exc}")

                if source_length > 0:
                    analysis_default = min(
                        168 if selected_input_source == INFLUENCE_SOURCE_CONTEXT else int(prediction_length),
                        source_length,
                    )
                    window_default = (
                        min(24, analysis_default)
                        if selected_input_source == INFLUENCE_SOURCE_CONTEXT
                        else max(1, analysis_default)
                    )

                    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
                    analysis_length = opt_col1.number_input(
                        "분석 입력 길이",
                        min_value=1,
                        max_value=source_length,
                        value=int(max(1, analysis_default)),
                        step=1,
                        key=f"influence_analysis_length_{source_key}",
                    )
                    window_size = opt_col2.number_input(
                        "Perturb window",
                        min_value=1,
                        max_value=int(analysis_length),
                        value=int(max(1, min(window_default, int(analysis_length)))),
                        step=1,
                        key=f"influence_window_size_{source_key}",
                    )
                    stride = opt_col3.number_input(
                        "Window 간격",
                        min_value=1,
                        max_value=int(analysis_length),
                        value=int(max(1, min(int(window_size), int(analysis_length)))),
                        step=1,
                        key=f"influence_stride_{source_key}",
                    )
                    max_windows = opt_col4.number_input(
                        "최대 window 수",
                        min_value=1,
                        max_value=100,
                        value=12,
                        step=1,
                        key=f"influence_max_windows_{source_key}",
                    )

                    replacement_method = st.selectbox(
                        "Perturb 방식",
                        options=["mean", "zero"],
                        format_func=lambda value: "평균/최빈값 대체" if value == "mean" else "0 대체(숫자형)",
                        key=f"influence_replacement_method_{source_key}",
                    )

                    preview_windows = build_influence_windows(
                        source_df=source_df,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        selected_id=selected_influence_id,
                        input_source=selected_input_source,
                        analysis_length=int(analysis_length),
                        window_size=int(window_size),
                        stride=int(stride),
                        max_windows=int(max_windows),
                    )
                    run_count_col1, run_count_col2, run_count_col3 = st.columns(3)
                    run_count_col1.metric("분석 window 수", int(len(preview_windows)))
                    run_count_col2.metric("예상 예측 호출", int(len(preview_windows) + 1))
                    run_count_col3.metric("예측 horizon", int(prediction_length))

                    if st.button("Influence map 계산", type="primary", disabled=not preview_windows):
                        try:
                            with st.spinner("Perturbation 기반 influence map을 계산하는 중입니다..."):
                                pipeline = get_pipeline(model_id, device)
                                influence_df, baseline_forecast_df = run_prediction_influence_analysis(
                                    pipeline=pipeline,
                                    context_df=influence_context_df,
                                    future_df=influence_future_df,
                                    prediction_length=int(prediction_length),
                                    id_column=id_column,
                                    timestamp_column=timestamp_column,
                                    target_column=target_column,
                                    selected_id=selected_influence_id,
                                    input_source=selected_input_source,
                                    perturb_column=perturb_column,
                                    analysis_length=int(analysis_length),
                                    window_size=int(window_size),
                                    stride=int(stride),
                                    max_windows=int(max_windows),
                                    replacement_method=replacement_method,
                                )
                                st.session_state["influence_df"] = influence_df
                                st.session_state["influence_baseline_forecast_df"] = baseline_forecast_df
                                st.session_state["influence_meta"] = {
                                    "series": selected_influence_id,
                                    "source": selected_input_source,
                                    "column": perturb_column,
                                    "windows": int(len(preview_windows)),
                                    "prediction_length": int(prediction_length),
                                    "replacement_method": replacement_method,
                                }
                            st.success("Influence map 계산이 완료되었습니다.")
                        except Exception as exc:
                            st.error(f"Influence map 계산 중 오류가 발생했습니다: {exc}")

                    influence_df = st.session_state.get("influence_df")
                    influence_meta = st.session_state.get("influence_meta")
                    if isinstance(influence_df, pd.DataFrame) and not influence_df.empty and influence_meta:
                        st.markdown("**Influence Map 결과**")
                        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                        meta_col1.metric("시계열", str(influence_meta["series"]))
                        meta_col2.metric("입력 소스", str(influence_meta["source"]))
                        meta_col3.metric("컬럼", str(influence_meta["column"]))
                        meta_col4.metric("window 수", int(influence_meta["windows"]))

                        value_column = st.radio(
                            "Heatmap 값",
                            options=["abs_delta", "delta"],
                            horizontal=True,
                            format_func=lambda value: "변화량 크기" if value == "abs_delta" else "부호 있는 변화량",
                            key="influence_heatmap_value",
                        )
                        st.plotly_chart(
                            build_influence_heatmap(influence_df=influence_df, value_column=value_column),
                            use_container_width=True,
                        )

                        influence_summary_df = summarize_influence_windows(influence_df)
                        result_col1, result_col2 = st.columns([1.0, 1.2])
                        with result_col1:
                            st.markdown("**상위 영향 window**")
                            st.dataframe(influence_summary_df.head(20), use_container_width=True)
                        with result_col2:
                            selected_window_label = st.selectbox(
                                "예측 변화 비교 window",
                                options=influence_summary_df["window_label"].tolist(),
                                key="influence_comparison_window",
                            )
                            st.plotly_chart(
                                build_influence_forecast_comparison(
                                    influence_df=influence_df,
                                    selected_window_label=selected_window_label,
                                ),
                                use_container_width=True,
                            )

                        with st.expander("Cell-level influence 데이터"):
                            st.dataframe(influence_df, use_container_width=True)
