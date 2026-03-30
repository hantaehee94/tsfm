from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from chronos2_core import (
    build_example_frames,
    detect_device,
    load_pipeline,
    load_table,
    run_prediction,
)


st.set_page_config(page_title="Chronos-2 Local Lab", layout="wide")
st.title("Chronos-2 Local Lab")
st.caption("맥북에서 로컬로 Chronos-2 예측 실험을 빠르게 돌리기 위한 최소 GUI")


@st.cache_resource(show_spinner=False)
def get_pipeline(model_id: str, device: str):
    return load_pipeline(model_id, device)


def show_series_preview(
    context_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
    target_column: str,
) -> None:
    if pred_df.empty:
        return

    available_ids = context_df[id_column].astype(str).unique().tolist()
    selected_id = st.selectbox("미리볼 시계열", options=available_ids)

    history = context_df.loc[context_df[id_column].astype(str) == selected_id, [timestamp_column, target_column]].copy()
    history = history.rename(columns={target_column: "history"}).set_index(timestamp_column)

    pred_cols = []
    for candidate in [0.1, 0.5, 0.9, "0.1", "0.5", "0.9", "median", "mean"]:
        if candidate in pred_df.columns:
            pred_cols.append(candidate)
    forecast = pred_df.loc[pred_df[id_column].astype(str) == selected_id, [timestamp_column, *pred_cols]].copy()
    forecast = forecast.set_index(timestamp_column)

    chart_df = history.join(forecast, how="outer")
    st.line_chart(chart_df)


def save_predictions_download(pred_df: pd.DataFrame) -> None:
    buffer = BytesIO()
    pred_df.to_csv(buffer, index=False)
    st.download_button(
        label="예측 결과 CSV 다운로드",
        data=buffer.getvalue(),
        file_name="chronos2_predictions.csv",
        mime="text/csv",
    )


with st.sidebar:
    st.header("실험 설정")
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
    prediction_length = st.number_input("예측 길이", min_value=1, max_value=512, value=24, step=1)
    source_mode = st.radio("데이터 소스", options=["합성 예제", "파일 업로드"], index=0)


if source_mode == "합성 예제":
    st.subheader("합성 데이터 실험")
    col1, col2 = st.columns(2)
    with col1:
        num_series = st.number_input("시계열 개수", min_value=1, max_value=50, value=3, step=1)
    with col2:
        context_length = st.number_input("과거 길이", min_value=8, max_value=2048, value=96, step=8)

    context_df, future_df = build_example_frames(
        num_series=int(num_series),
        context_length=int(context_length),
        prediction_length=int(prediction_length),
    )
    id_column = "id"
    timestamp_column = "timestamp"
    target_column = "target"

    st.markdown("`run_forecast.py`와 같은 합성 데이터로 바로 예측을 실행합니다.")
else:
    st.subheader("파일 업로드 실험")
    st.markdown("`context_df`와 `future_df`를 각각 업로드하면 됩니다. 지원 형식은 CSV, Parquet입니다.")

    context_file = st.file_uploader("과거 데이터 context_df", type=["csv", "parquet"])
    future_file = st.file_uploader("미래 공변량 future_df", type=["csv", "parquet"])

    context_df = pd.DataFrame()
    future_df = pd.DataFrame()
    id_column = ""
    timestamp_column = ""
    target_column = ""

    if context_file and future_file:
        context_df = load_table(context_file.name, context_file.getvalue())
        future_df = load_table(future_file.name, future_file.getvalue())
        common_columns = [col for col in context_df.columns if col in future_df.columns]

        meta1, meta2, meta3 = st.columns(3)
        with meta1:
            id_column = st.selectbox("id 컬럼", options=context_df.columns, index=context_df.columns.get_loc(common_columns[0]) if common_columns else 0)
        with meta2:
            timestamp_column = st.selectbox(
                "timestamp 컬럼",
                options=context_df.columns,
                index=context_df.columns.get_loc("timestamp") if "timestamp" in context_df.columns else 0,
            )
        with meta3:
            target_column = st.selectbox(
                "target 컬럼",
                options=context_df.columns,
                index=context_df.columns.get_loc("target") if "target" in context_df.columns else 0,
            )

if not context_df.empty:
    preview1, preview2 = st.columns(2)
    with preview1:
        st.markdown("**Context Preview**")
        st.dataframe(context_df.head(10), use_container_width=True)
    with preview2:
        st.markdown("**Future Preview**")
        st.dataframe(future_df.head(10), use_container_width=True)

can_run = not context_df.empty and not future_df.empty

if st.button("Chronos-2 예측 실행", type="primary", disabled=not can_run):
    try:
        with st.spinner("모델을 불러오고 예측하는 중입니다..."):
            pipeline = get_pipeline(model_id, device)
            st.session_state["pred_df"] = run_prediction(
                pipeline=pipeline,
                context_df=context_df,
                future_df=future_df,
                prediction_length=int(prediction_length),
                id_column=id_column,
                timestamp_column=timestamp_column,
                target_column=target_column,
            )
            st.session_state["result_meta"] = {
                "id_column": id_column,
                "timestamp_column": timestamp_column,
                "target_column": target_column,
            }

        st.success("예측이 완료되었습니다.")
    except Exception as exc:
        st.error(f"실행 중 오류가 발생했습니다: {exc}")

pred_df = st.session_state.get("pred_df")
result_meta = st.session_state.get("result_meta")

if isinstance(pred_df, pd.DataFrame) and not pred_df.empty and result_meta:
    st.success("예측 결과가 준비되어 있습니다.")
    st.dataframe(pred_df.head(50), use_container_width=True)
    show_series_preview(
        context_df=context_df,
        pred_df=pred_df,
        id_column=result_meta["id_column"],
        timestamp_column=result_meta["timestamp_column"],
        target_column=result_meta["target_column"],
    )
    save_predictions_download(pred_df)
