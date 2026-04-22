from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@dataclass
class TSFSeriesRecord:
    series_id: str
    start_timestamp: pd.Timestamp | None
    values: np.ndarray
    attributes: dict[str, object]


@st.cache_data(show_spinner=False)
def load_tsf_summary(uploaded_file_name: str, raw_bytes: bytes) -> dict[str, object]:
    text, encoding = decode_tsf_bytes(raw_bytes)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    attributes: list[tuple[str, str]] = []
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    preview_lines: list[str] = []
    in_data_section = False

    for line in lines:
        if line.startswith("#"):
            continue
        lower_line = line.lower()
        if not in_data_section and len(preview_lines) < 20:
            preview_lines.append(line)
        if lower_line.startswith("@attribute"):
            parts = line.split()
            if len(parts) < 3:
                raise ValueError("TSF의 @attribute 형식이 올바르지 않습니다.")
            attributes.append((parts[1], parts[2].lower()))
            continue
        if lower_line == "@data":
            in_data_section = True
            continue
        if in_data_section:
            data_lines.append(line)
            continue
        if line.startswith("@"):
            key, _, value = line[1:].partition(" ")
            metadata[key.lower()] = value.strip()

    if not data_lines:
        raise ValueError("TSF 파일에서 @data 구간을 찾지 못했습니다.")

    series_records: list[TSFSeriesRecord] = []
    summary_rows: list[dict[str, object]] = []
    frequency = metadata.get("frequency", "")

    for series_idx, line in enumerate(data_lines, start=1):
        parts = [part.strip() for part in line.split(":")]
        if len(parts) != len(attributes) + 1:
            raise ValueError(
                f"TSF 데이터 행 형식이 맞지 않습니다. series_index={series_idx}, expected_parts={len(attributes) + 1}, found={len(parts)}"
            )

        attribute_values: dict[str, object] = {}
        for (attr_name, attr_type), raw_value in zip(attributes, parts[:-1], strict=False):
            attribute_values[attr_name] = parse_tsf_attribute_value(raw_value, attr_type)

        values = np.array([parse_tsf_series_value(token) for token in parts[-1].split(",")], dtype="float64")
        start_timestamp = resolve_tsf_start_timestamp(attribute_values)
        series_id = build_tsf_series_id(attribute_values, series_idx)
        non_missing = values[np.isfinite(values)]

        series_records.append(
            TSFSeriesRecord(
                series_id=series_id,
                start_timestamp=start_timestamp,
                values=values,
                attributes=attribute_values,
            )
        )
        summary_rows.append(
            {
                "series_id": series_id,
                "length": int(values.size),
                "missing_count": int(np.isnan(values).sum()),
                "missing_ratio": float(np.isnan(values).mean()) if values.size else np.nan,
                "min": float(np.min(non_missing)) if non_missing.size else np.nan,
                "max": float(np.max(non_missing)) if non_missing.size else np.nan,
                "mean": float(np.mean(non_missing)) if non_missing.size else np.nan,
                "std": float(np.std(non_missing)) if non_missing.size else np.nan,
                "start_timestamp": start_timestamp,
                **attribute_values,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    attribute_df = pd.DataFrame(attributes, columns=["attribute", "type"])

    total_points = int(sum(len(record.values) for record in series_records))
    total_missing = int(sum(np.isnan(record.values).sum() for record in series_records))

    return {
        "file_name": uploaded_file_name,
        "encoding": encoding,
        "metadata": metadata,
        "attributes": attribute_df,
        "series_records": series_records,
        "summary_df": summary_df,
        "preview_lines": preview_lines,
        "total_series": int(len(series_records)),
        "total_points": total_points,
        "total_missing": total_missing,
        "frequency": frequency,
    }


def decode_tsf_bytes(raw_bytes: bytes) -> tuple[str, str]:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("TSF 파일을 디코딩할 수 없습니다.")


def parse_tsf_attribute_value(raw_value: str, attr_type: str) -> object:
    value = raw_value.strip()
    if value == "?":
        return np.nan
    if attr_type in {"numeric", "real", "integer"}:
        return float(value)
    if attr_type == "date":
        return pd.to_datetime(value, errors="coerce")
    return value


def parse_tsf_series_value(token: str) -> float:
    value = token.strip()
    if value == "?":
        return float("nan")
    return float(value)


def resolve_tsf_start_timestamp(attribute_values: dict[str, object]) -> pd.Timestamp | None:
    for key, value in attribute_values.items():
        if "start" in key.lower() and isinstance(value, pd.Timestamp):
            return value
    date_values = [value for value in attribute_values.values() if isinstance(value, pd.Timestamp)]
    if date_values:
        return date_values[0]
    return None


def build_tsf_series_id(attribute_values: dict[str, object], series_idx: int) -> str:
    preferred_keys = ["id", "series_name", "series_id"]
    lowered = {key.lower(): key for key in attribute_values}
    for key in preferred_keys:
        if key in lowered:
            return str(attribute_values[lowered[key]])

    pieces = [str(value) for value in attribute_values.values() if not isinstance(value, pd.Timestamp)]
    if pieces:
        return "__".join(pieces)
    return f"series_{series_idx:04d}"


def tsf_frequency_to_offset(frequency: str):
    normalized = frequency.strip().lower()
    mapping = {
        "yearly": pd.DateOffset(years=1),
        "quarterly": pd.DateOffset(months=3),
        "monthly": pd.DateOffset(months=1),
        "weekly": pd.DateOffset(weeks=1),
        "daily": pd.DateOffset(days=1),
        "hourly": pd.DateOffset(hours=1),
        "half_hourly": pd.DateOffset(minutes=30),
        "half-hourly": pd.DateOffset(minutes=30),
        "quarter_hourly": pd.DateOffset(minutes=15),
        "quarter-hourly": pd.DateOffset(minutes=15),
        "minutely": pd.DateOffset(minutes=1),
    }
    return mapping.get(normalized)


def build_series_frame(record: TSFSeriesRecord, frequency: str) -> pd.DataFrame:
    offset = tsf_frequency_to_offset(frequency)
    base_timestamp = record.start_timestamp if record.start_timestamp is not None else pd.Timestamp("1970-01-01")
    timestamps: list[pd.Timestamp] = []
    for idx in range(len(record.values)):
        if offset is None:
            timestamps.append(base_timestamp + pd.to_timedelta(idx, unit="D"))
        else:
            timestamps.append(base_timestamp + idx * offset)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": record.values,
            "is_missing": np.isnan(record.values),
            "step": np.arange(len(record.values)),
        }
    )


def build_length_histogram(summary_df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=summary_df["length"],
            marker={"color": "#264653"},
            nbinsx=min(50, max(10, int(np.sqrt(max(len(summary_df), 1))))),
            name="series_length",
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis_title="Series Length",
        yaxis_title="Count",
        bargap=0.05,
    )
    return figure


def build_missing_scatter(summary_df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=summary_df["length"],
            y=summary_df["missing_ratio"],
            mode="markers",
            marker={"size": 8, "color": "#e76f51", "opacity": 0.7},
            text=summary_df["series_id"],
            name="series",
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis_title="Series Length",
        yaxis_title="Missing Ratio",
    )
    return figure


def build_series_plot(series_df: pd.DataFrame, series_id: str) -> go.Figure:
    figure = go.Figure()
    observed_df = series_df.loc[~series_df["is_missing"]].copy()
    missing_df = series_df.loc[series_df["is_missing"]].copy()

    figure.add_trace(
        go.Scattergl(
            x=observed_df["timestamp"],
            y=observed_df["value"],
            mode="lines",
            name="observed",
            line={"color": "#1d3557", "width": 2},
        )
    )
    if not missing_df.empty:
        figure.add_trace(
            go.Scattergl(
                x=missing_df["timestamp"],
                y=[None] * len(missing_df),
                mode="markers",
                name="missing",
                marker={"color": "#e63946", "size": 7, "symbol": "x"},
                text=[f"missing at step {step}" for step in missing_df["step"]],
            )
        )

    figure.update_layout(
        title=f"Series Preview: {series_id}",
        height=420,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        legend={"orientation": "h"},
        xaxis_title="Timestamp",
        yaxis_title="Value",
    )
    return figure


st.set_page_config(page_title="TSF Explorer", layout="wide")
st.title("TSF Explorer")
st.caption("대용량 TSF 파일의 메타데이터, 시계열 구조, 결측, 샘플 시계열을 빠르게 확인하는 전용 뷰어")

with st.sidebar:
    st.header("안내")
    st.markdown(
        "- TSF 파일 하나를 업로드해서 구조를 먼저 파악합니다.\n"
        "- 전체를 long-format으로 모두 펼치지 않고 요약 중심으로 봅니다.\n"
        "- 큰 파일일수록 첫 파싱은 시간이 걸릴 수 있지만 이후에는 캐시를 재사용합니다."
    )

uploaded_file = st.file_uploader("TSF 파일 업로드", type=["tsf"])

if uploaded_file is None:
    st.info("TSF 파일을 올리면 메타데이터, 속성, 시계열 길이 분포, 결측 현황, 샘플 시계열을 확인할 수 있습니다.")
    st.stop()

with st.spinner("TSF 파일을 읽는 중입니다..."):
    tsf_data = load_tsf_summary(uploaded_file.name, uploaded_file.getvalue())

summary_df = tsf_data["summary_df"]
attribute_df = tsf_data["attributes"]
metadata = tsf_data["metadata"]
series_records = tsf_data["series_records"]

top1, top2, top3, top4 = st.columns(4)
top1.metric("파일명", tsf_data["file_name"])
top2.metric("인코딩", tsf_data["encoding"])
top3.metric("시계열 수", int(tsf_data["total_series"]))
top4.metric("전체 포인트 수", int(tsf_data["total_points"]))

info1, info2, info3, info4 = st.columns(4)
info1.metric("결측 수", int(tsf_data["total_missing"]))
info2.metric("결측 비율", f"{(tsf_data['total_missing'] / max(tsf_data['total_points'], 1)) * 100:.2f}%")
info3.metric("최소 길이", int(summary_df["length"].min()) if not summary_df.empty else 0)
info4.metric("최대 길이", int(summary_df["length"].max()) if not summary_df.empty else 0)

meta_col1, meta_col2 = st.columns([0.9, 1.1])
with meta_col1:
    with st.container(border=True):
        st.markdown("**파일 메타데이터**")
        if metadata:
            meta_df = pd.DataFrame(
                [{"key": key, "value": value} for key, value in metadata.items()]
            )
            st.dataframe(meta_df, use_container_width=True, height=260)
        else:
            st.info("메타데이터가 거의 없거나 파악 가능한 항목이 없습니다.")
with meta_col2:
    with st.container(border=True):
        st.markdown("**속성 스키마**")
        if not attribute_df.empty:
            st.dataframe(attribute_df, use_container_width=True, height=260)
        else:
            st.info("@attribute 정의가 없습니다.")

dist_col1, dist_col2 = st.columns(2)
with dist_col1:
    with st.container(border=True):
        st.markdown("**시계열 길이 분포**")
        st.plotly_chart(build_length_histogram(summary_df), use_container_width=True)
with dist_col2:
    with st.container(border=True):
        st.markdown("**길이 대비 결측 비율**")
        st.plotly_chart(build_missing_scatter(summary_df), use_container_width=True)

detail_col1, detail_col2 = st.columns([1.2, 0.8])
with detail_col1:
    with st.container(border=True):
        st.markdown("**시계열 요약 테이블**")
        st.dataframe(summary_df, use_container_width=True, height=360)
with detail_col2:
    with st.container(border=True):
        st.markdown("**원본 헤더 미리보기**")
        st.code("\n".join(tsf_data["preview_lines"]), language="text")

with st.container(border=True):
    st.markdown("**샘플 시계열 탐색**")
    preview_col1, preview_col2 = st.columns([0.6, 0.4])
    sort_option = preview_col2.selectbox(
        "시계열 정렬 기준",
        options=["원본 순서", "길이 긴 순", "결측 많은 순"],
        index=0,
    )

    if sort_option == "길이 긴 순":
        sorted_df = summary_df.sort_values(["length", "series_id"], ascending=[False, True])
    elif sort_option == "결측 많은 순":
        sorted_df = summary_df.sort_values(["missing_count", "series_id"], ascending=[False, True])
    else:
        sorted_df = summary_df.copy()

    series_options = sorted_df["series_id"].astype(str).tolist()
    selected_series_id = preview_col1.selectbox("시계열 선택", options=series_options, index=0)

    selected_record = next(record for record in series_records if record.series_id == selected_series_id)
    selected_series_df = build_series_frame(selected_record, tsf_data["frequency"])
    st.plotly_chart(build_series_plot(selected_series_df, selected_series_id), use_container_width=True)

    attr_rows = [{"attribute": key, "value": value} for key, value in selected_record.attributes.items()]
    attr_rows.append({"attribute": "derived_start_timestamp", "value": selected_record.start_timestamp})
    attr_rows.append({"attribute": "length", "value": int(len(selected_record.values))})
    attr_rows.append({"attribute": "missing_count", "value": int(np.isnan(selected_record.values).sum())})
    st.dataframe(pd.DataFrame(attr_rows), use_container_width=True, height=260)
