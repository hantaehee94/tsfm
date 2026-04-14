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
    build_example_frames,
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


@st.cache_resource(show_spinner=False)
def get_pipeline(model_id: str, device: str):
    return load_pipeline(model_id, device)


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
    past_covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    keep_context_columns = [id_column, timestamp_column, target_column, *past_covariates]
    prepared_context = context_df.loc[:, [col for col in keep_context_columns if col in context_df.columns]].copy()
    prepared_full_context = full_context_df.loc[:, [col for col in keep_context_columns if col in full_context_df.columns]].copy()

    if future_df is None or future_df.empty:
        return prepared_context, prepared_full_context, future_df

    keep_future_columns = [col for col in future_df.columns if col not in past_covariates]
    prepared_future = future_df.loc[:, keep_future_columns].copy()
    return prepared_context, prepared_full_context, prepared_future


def add_weekday_covariate(
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    id_column: str,
    timestamp_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    prepared_context = context_df.copy()
    prepared_context[timestamp_column] = pd.to_datetime(prepared_context[timestamp_column], errors="coerce")
    prepared_context[WEEKDAY_COVARIATE_COLUMN] = prepared_context[timestamp_column].dt.day_name()

    if future_df is not None and not future_df.empty:
        prepared_future = future_df.copy()
        prepared_future[timestamp_column] = pd.to_datetime(prepared_future[timestamp_column], errors="coerce")
        prepared_future[WEEKDAY_COVARIATE_COLUMN] = prepared_future[timestamp_column].dt.day_name()
        return prepared_context, prepared_future

    derived_future = prepared_context[[id_column, timestamp_column]].drop_duplicates().copy()
    derived_future[WEEKDAY_COVARIATE_COLUMN] = derived_future[timestamp_column].dt.day_name()
    return prepared_context, derived_future


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
        return None, None

    if future_df is None or future_df.empty:
        return actual_future, None

    prepared_future = future_df.copy()
    prepared_future[timestamp_column] = pd.to_datetime(prepared_future[timestamp_column], errors="coerce")
    horizon_keys = actual_future[[id_column, timestamp_column]].drop_duplicates()
    filtered_future = prepared_future.merge(horizon_keys, on=[id_column, timestamp_column], how="inner")
    return actual_future, filtered_future


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
    if lengths.min() < context_length + prediction_length:
        raise ValueError(
            "선택한 데이터 길이가 부족합니다. "
            f"각 시계열은 최소 {context_length + prediction_length}개 이상이어야 합니다."
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return summary_df, windows_df


def build_validation_comparison_summary(validation_windows_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["MAE", "RMSE", "MAPE", "MASE", "Correlation"]
    if validation_windows_df.empty or "scenario" not in validation_windows_df.columns:
        return pd.DataFrame()

    grouped = validation_windows_df.groupby("scenario", dropna=False)[metric_columns].mean().reset_index()
    if set(grouped["scenario"]) != {"요일 미사용", "요일 사용"}:
        return grouped

    baseline = grouped.loc[grouped["scenario"] == "요일 미사용"].iloc[0]
    weekday = grouped.loc[grouped["scenario"] == "요일 사용"].iloc[0]
    comparison_rows: list[dict[str, object]] = []

    for metric in metric_columns:
        delta = float(weekday[metric] - baseline[metric])
        if metric == "Correlation":
            better = "개선" if delta > 0 else "악화" if delta < 0 else "동일"
        else:
            better = "개선" if delta < 0 else "악화" if delta > 0 else "동일"
        comparison_rows.append(
            {
                "metric": metric,
                "요일 미사용": float(baseline[metric]),
                "요일 사용": float(weekday[metric]),
                "delta": delta,
                "판정": better,
            }
        )

    return pd.DataFrame(comparison_rows)


with st.sidebar:
    st.header("실험 설정")
    page_mode = st.radio(
        "작업 화면",
        options=["단일 예측", "자동 검증", "공변량 분석"],
        index=0,
    )
    model_id = st.text_input("모델 ID", value="amazon/chronos-2")
    device_options = []
    for candidate in [detect_device(), "cpu", "mps", "cuda"]:
        if candidate not in device_options:
            device_options.append(candidate)
    device = st.selectbox(
        "실행 장치",
        options=device_options,
        index=0,
    )
    prediction_length = st.number_input(
        "예측 길이",
        min_value=1,
        max_value=CHRONOS2_MAX_PREDICTION_LENGTH,
        value=24,
        step=1,
    )
    source_mode = st.radio("데이터 소스", options=["합성 예제", "파일 업로드"], index=0)

context_df = pd.DataFrame()
full_context_df = pd.DataFrame()
future_df = None
id_column = ""
timestamp_column = ""
target_column = ""
experiment_mode = "선택 구간 끝에서 미래 예측"

with st.expander("데이터 준비", expanded=True):
    if source_mode == "합성 예제":
        st.subheader("합성 데이터 실험")
        col1, col2 = st.columns(2)
        with col1:
            num_series = st.number_input("시계열 개수", min_value=1, max_value=50, value=3, step=1)
        with col2:
            context_length = st.number_input(
                "과거 길이",
                min_value=8,
                max_value=CHRONOS2_MAX_CONTEXT_LENGTH,
                value=96,
                step=8,
            )
        use_future_covariates = st.checkbox("미래 공변량 사용", value=True)

        context_df, generated_future_df = build_example_frames(
            num_series=int(num_series),
            context_length=int(context_length),
            prediction_length=int(prediction_length),
        )
        full_context_df = context_df.copy()
        future_df = generated_future_df if use_future_covariates else None
        id_column = "id"
        timestamp_column = "timestamp"
        target_column = "target"

        if use_future_covariates:
            st.markdown("`run_forecast.py`와 같은 합성 데이터와 미래 공변량으로 예측을 실행합니다.")
        else:
            st.markdown("합성 데이터의 과거 구간만 사용해서, 미래 공변량 없이 예측을 실행합니다.")
    else:
        st.subheader("파일 업로드 실험")
        st.markdown("`context_df`는 필수이고, `future_df`는 선택입니다. 지원 형식은 CSV, Parquet입니다.")

        context_file = st.file_uploader("과거 데이터 context_df", type=["csv", "parquet"])
        future_file = st.file_uploader("미래 공변량 future_df (선택)", type=["csv", "parquet"])

        if context_file:
            context_df = load_table(context_file.name, context_file.getvalue())
            full_context_df = context_df.copy()
            if future_file:
                future_df = load_table(future_file.name, future_file.getvalue())
            guessed_id_column = guess_id_column(context_df)
            guessed_timestamp_column = guess_timestamp_column(context_df)
            guessed_target_column = guess_target_column(context_df)

            id_options = [AUTO_ID_OPTION, *context_df.columns.tolist()]
            id_index = 0 if guessed_id_column == AUTO_ID_OPTION else context_df.columns.get_loc(guessed_id_column) + 1

            meta1, meta2, meta3 = st.columns(3)
            with meta1:
                id_column = st.selectbox(
                    "id 컬럼",
                    options=id_options,
                    index=id_index,
                    format_func=lambda value: "단일 시계열로 자동 생성" if value == AUTO_ID_OPTION else value,
                )
            with meta2:
                timestamp_column = st.selectbox(
                    "timestamp 컬럼",
                    options=context_df.columns,
                    index=context_df.columns.get_loc(guessed_timestamp_column),
                )
            with meta3:
                target_column = st.selectbox(
                    "target 컬럼",
                    options=context_df.columns,
                    index=context_df.columns.get_loc(guessed_target_column),
                )

            context_df, future_df, id_column = apply_id_selection(
                context_df=context_df,
                future_df=future_df,
                id_selection=id_column,
            )
            full_context_df = context_df.copy()

            available_past_covariates = [
                col
                for col in context_df.columns
                if col not in {id_column, timestamp_column, target_column}
            ]
            selected_past_covariates = st.multiselect(
                "past covariates로 사용할 컬럼",
                options=available_past_covariates,
                default=[],
            )
            context_df, full_context_df, future_df = filter_model_columns(
                context_df=context_df,
                full_context_df=full_context_df,
                future_df=future_df,
                id_column=id_column,
                timestamp_column=timestamp_column,
                target_column=target_column,
                past_covariates=selected_past_covariates,
            )

            lengths = get_series_lengths(context_df, id_column=id_column, timestamp_column=timestamp_column)
            inspect_ids = context_df[id_column].astype(str).drop_duplicates().tolist()
            inspect_id = st.selectbox("구간 확인용 시계열", options=inspect_ids)
            inspect_timestamps = get_index_timestamp_map(
                df=context_df,
                id_column=id_column,
                timestamp_column=timestamp_column,
                selected_id=inspect_id,
            )

            st.markdown("**타임스탬프 요약**")
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
            ts_col1.info(f"시작 타임스탬프: {inspect_timestamps.iloc[int(start_idx)]}")
            ts_col2.info(f"종료 타임스탬프: {inspect_timestamps.iloc[int(end_idx)]}")

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

actual_future_df = None

if not context_df.empty:
    series_lengths = get_series_lengths(context_df, id_column=id_column, timestamp_column=timestamp_column)
    if int(series_lengths.max()) > CHRONOS2_MAX_CONTEXT_LENGTH:
        st.warning(
            "Chronos-2 공식 스펙에 맞추기 위해 각 시계열의 최근 "
            f"{CHRONOS2_MAX_CONTEXT_LENGTH}개 시점만 모델 입력에 사용합니다."
        )
    preview1, preview2 = st.columns(2)
    with preview1:
        show_context_summary(
            context_df=context_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )
        st.markdown("**Context Preview**")
        st.dataframe(context_df.head(10), use_container_width=True)
    with preview2:
        st.markdown("**Future Preview**")
        if future_df is None or future_df.empty:
            st.info("미래 공변량 없이 실행합니다.")
        else:
            st.dataframe(future_df.head(10), use_container_width=True)

can_run = not context_df.empty

if page_mode == "단일 예측":
    st.subheader("단일 예측")
    if st.button("Chronos-2 예측 실행", type="primary", disabled=not can_run):
        try:
            with st.spinner("모델을 불러오고 예측하는 중입니다..."):
                pipeline = get_pipeline(model_id, device)
                model_context_df = trim_context_to_model_limit(
                    context_df=context_df,
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                )
                model_future_df = future_df
                actual_future_df = None

                if source_mode == "파일 업로드" and experiment_mode == "데이터셋 내부 평가":
                    model_context_df, actual_future_df, model_future_df = build_evaluation_split(
                        context_df=context_df,
                        future_df=future_df,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        prediction_length=int(prediction_length),
                    )
                elif source_mode == "파일 업로드" and experiment_mode == "선택 구간 끝에서 미래 예측":
                    actual_future_df, model_future_df = build_future_comparison_split(
                        full_context_df=full_context_df,
                        model_context_df=model_context_df,
                        future_df=future_df,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        prediction_length=int(prediction_length),
                    )

                model_context_df = trim_context_to_model_limit(
                    context_df=model_context_df,
                    id_column=id_column,
                    timestamp_column=timestamp_column,
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

            st.success("예측이 완료되었습니다.")
        except Exception as exc:
            st.error(f"실행 중 오류가 발생했습니다: {exc}")

    pred_df = st.session_state.get("pred_df")
    result_meta = st.session_state.get("result_meta")
    metrics_df = st.session_state.get("metrics_df")

    if isinstance(pred_df, pd.DataFrame) and not pred_df.empty and result_meta:
        st.success("예측 결과가 준비되어 있습니다.")
        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            st.markdown("**평가 지표**")
            st.dataframe(metrics_df, use_container_width=True)
        show_model_input_summary(
            context_df=result_meta["history_df"],
            future_df=result_meta["model_future_df"],
            id_column=result_meta["id_column"],
            timestamp_column=result_meta["timestamp_column"],
            target_column=result_meta["target_column"],
        )
        st.dataframe(pred_df.head(50), use_container_width=True)
        show_series_preview(
            context_df=result_meta["history_df"],
            pred_df=pred_df,
            actual_future_df=result_meta["actual_future_df"],
            id_column=result_meta["id_column"],
            timestamp_column=result_meta["timestamp_column"],
            target_column=result_meta["target_column"],
        )
        save_predictions_download(pred_df)

elif page_mode == "자동 검증":
    st.subheader("자동 검증")
    st.caption("현재 준비된 데이터 구간을 대상으로 슬라이딩 윈도우 평가를 수행합니다.")
    val_col1, val_col2, val_col3 = st.columns(3)
    with val_col1:
        validation_context_length = st.number_input(
            "검증 context length",
            min_value=8,
            max_value=CHRONOS2_MAX_CONTEXT_LENGTH,
            value=min(168, CHRONOS2_MAX_CONTEXT_LENGTH),
            step=8,
        )
    with val_col2:
        validation_prediction_length = st.number_input(
            "검증 prediction length",
            min_value=1,
            max_value=CHRONOS2_MAX_PREDICTION_LENGTH,
            value=int(prediction_length),
            step=1,
        )
    with val_col3:
        validation_stride = st.number_input(
            "stride",
            min_value=1,
            max_value=CHRONOS2_MAX_CONTEXT_LENGTH,
            value=1,
            step=1,
        )

    max_windows_input = st.number_input(
        "최대 window 수 (0이면 전체)",
        min_value=0,
        max_value=10000,
        value=0,
        step=1,
    )
    max_windows = None if int(max_windows_input) == 0 else int(max_windows_input)
    compare_weekday_covariate = st.checkbox(
        "요일 categorical covariate A/B 비교",
        value=False,
        help="timestamp에서 요일을 자동 생성해 `요일 미사용`과 `요일 사용` 검증 결과를 함께 비교합니다.",
    )

    if st.button("슬라이딩 윈도우 검증 실행", type="primary", disabled=not can_run):
        try:
            with st.spinner("슬라이딩 윈도우 검증을 수행하는 중입니다..."):
                pipeline = get_pipeline(model_id, device)
                summary_df, windows_df = run_sliding_window_validation(
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
                summary_df["scenario"] = "요일 미사용"
                windows_df["scenario"] = "요일 미사용"

                comparison_summary_df = pd.DataFrame()
                if compare_weekday_covariate:
                    weekday_context_df, weekday_future_df = add_weekday_covariate(
                        context_df=context_df,
                        future_df=future_df,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                    )
                    weekday_summary_df, weekday_windows_df = run_sliding_window_validation(
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
                    weekday_summary_df["scenario"] = "요일 사용"
                    weekday_windows_df["scenario"] = "요일 사용"
                    summary_df = pd.concat([summary_df, weekday_summary_df], ignore_index=True)
                    windows_df = pd.concat([windows_df, weekday_windows_df], ignore_index=True)
                    comparison_summary_df = build_validation_comparison_summary(windows_df)

                st.session_state["validation_summary_df"] = summary_df
                st.session_state["validation_windows_df"] = windows_df
                st.session_state["validation_comparison_summary_df"] = comparison_summary_df
            st.success("자동 검증이 완료되었습니다.")
        except Exception as exc:
            st.error(f"자동 검증 중 오류가 발생했습니다: {exc}")

    validation_summary_df = st.session_state.get("validation_summary_df")
    validation_windows_df = st.session_state.get("validation_windows_df")
    validation_comparison_summary_df = st.session_state.get("validation_comparison_summary_df")
    if isinstance(validation_summary_df, pd.DataFrame) and not validation_summary_df.empty:
        st.markdown("**검증 요약**")
        st.dataframe(validation_summary_df, use_container_width=True)
    if isinstance(validation_comparison_summary_df, pd.DataFrame) and not validation_comparison_summary_df.empty:
        st.markdown("**요일 covariate 비교**")
        st.dataframe(validation_comparison_summary_df, use_container_width=True)
    if isinstance(validation_windows_df, pd.DataFrame) and not validation_windows_df.empty:
        st.markdown("**window별 결과**")
        st.dataframe(validation_windows_df, use_container_width=True)
        metric_for_plot = st.selectbox(
            "추이로 볼 지표",
            options=["MAE", "RMSE", "MAPE", "MASE", "Correlation"],
            index=0,
        )
        trend_fig = go.Figure()
        if "scenario" in validation_windows_df.columns:
            line_colors = {"요일 미사용": "#1d3557", "요일 사용": "#e76f51"}
            for scenario_name, group in validation_windows_df.groupby("scenario", dropna=False):
                group = group.sort_values("window_index")
                trend_fig.add_trace(
                    go.Scatter(
                        x=group["window_index"],
                        y=group[metric_for_plot],
                        mode="lines+markers",
                        name=str(scenario_name),
                        line={"color": line_colors.get(str(scenario_name), "#457b9d"), "width": 2},
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

else:
    st.subheader("공변량 분석")
    st.info(
        "이 화면은 공변량 민감도 분석을 위한 자리입니다. "
        "다음 단계로는 선택한 covariate를 shuffle/mean-fix/scale 해서 "
        "원본 예측과 비교하는 counterfactual 분석을 붙이는 것이 가장 좋습니다."
    )
    if can_run:
        show_model_input_summary(
            context_df=trim_context_to_model_limit(
                context_df=context_df,
                id_column=id_column,
                timestamp_column=timestamp_column,
            ),
            future_df=future_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target_column=target_column,
        )
