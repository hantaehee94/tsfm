from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

CHRONOS2_MAX_CONTEXT_LENGTH = 8192
CHRONOS2_MAX_PREDICTION_LENGTH = 1024


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(model_id: str, device: str) -> Chronos2Pipeline:
    return Chronos2Pipeline.from_pretrained(model_id, device_map=device)


def validate_chronos2_lengths(
    prediction_length: int,
    context_length: int | None = None,
) -> None:
    if prediction_length < 1:
        raise ValueError("prediction_length는 1 이상이어야 합니다.")
    if prediction_length > CHRONOS2_MAX_PREDICTION_LENGTH:
        raise ValueError(
            f"Chronos-2 공식 스펙상 prediction_length는 최대 {CHRONOS2_MAX_PREDICTION_LENGTH}까지 지원합니다."
        )
    if context_length is not None and context_length < 1:
        raise ValueError("context_length는 1 이상이어야 합니다.")
    if context_length is not None and context_length > CHRONOS2_MAX_CONTEXT_LENGTH:
        raise ValueError(
            f"Chronos-2 공식 스펙상 context_length는 최대 {CHRONOS2_MAX_CONTEXT_LENGTH}까지 지원합니다."
        )


def trim_context_to_model_limit(
    context_df: pd.DataFrame,
    id_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    prepared = context_df.copy()
    prepared[timestamp_column] = pd.to_datetime(prepared[timestamp_column], errors="coerce")
    prepared = prepared.sort_values([id_column, timestamp_column])
    return prepared.groupby(id_column, group_keys=False).tail(CHRONOS2_MAX_CONTEXT_LENGTH)


def run_prediction(
    pipeline: Chronos2Pipeline,
    context_df: pd.DataFrame,
    future_df: pd.DataFrame | None,
    prediction_length: int,
    id_column: str,
    timestamp_column: str,
    target_column: str,
    quantile_levels: list[float] | None = None,
) -> pd.DataFrame:
    validate_chronos2_lengths(prediction_length=prediction_length)

    if quantile_levels is None:
        quantile_levels = [0.1, 0.5, 0.9]

    prepared_context = trim_context_to_model_limit(
        context_df=context_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    prepared_future = None
    if future_df is not None and not future_df.empty:
        prepared_future = future_df.copy()
        prepared_future[timestamp_column] = pd.to_datetime(prepared_future[timestamp_column])

    return pipeline.predict_df(
        prepared_context,
        future_df=prepared_future,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target=target_column,
    )


def parse_tsf(raw_bytes: bytes) -> pd.DataFrame:
    text = _decode_tsf_bytes(raw_bytes)
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]

    attributes: list[tuple[str, str]] = []
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    in_data_section = False

    for line in lines:
        lower_line = line.lower()
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
        raise ValueError("TSF 데이터 구간(@data)을 찾지 못했습니다.")

    frequency = metadata.get("frequency", "")
    rows: list[dict[str, object]] = []

    for series_idx, line in enumerate(data_lines, start=1):
        parts = [part.strip() for part in line.split(":")]
        if len(parts) != len(attributes) + 1:
            raise ValueError("TSF 데이터 행이 @attribute 정의와 맞지 않습니다.")

        attribute_values: dict[str, object] = {}
        for (attr_name, attr_type), raw_value in zip(attributes, parts[:-1], strict=False):
            attribute_values[attr_name] = _parse_tsf_attribute_value(raw_value, attr_type)

        series_values = [_parse_tsf_series_value(token) for token in parts[-1].split(",")]
        start_timestamp = _resolve_tsf_start_timestamp(attribute_values)
        series_id = _build_tsf_series_id(attribute_values, series_idx)

        for step, value in enumerate(series_values):
            rows.append(
                {
                    "id": series_id,
                    "timestamp": _build_tsf_timestamp(start_timestamp, step, frequency),
                    "target": value,
                    **attribute_values,
                }
            )

    return pd.DataFrame(rows)


def _decode_tsf_bytes(raw_bytes: bytes) -> str:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("TSF 파일을 디코딩할 수 없습니다.")


def _parse_tsf_attribute_value(raw_value: str, attr_type: str) -> object:
    value = raw_value.strip()
    if value == "?":
        return np.nan
    if attr_type in {"numeric", "real", "integer"}:
        return float(value)
    if attr_type == "date":
        return pd.to_datetime(value, errors="coerce")
    return value


def _parse_tsf_series_value(token: str) -> float:
    value = token.strip()
    if value == "?":
        return float("nan")
    return float(value)


def _resolve_tsf_start_timestamp(attribute_values: dict[str, object]) -> pd.Timestamp | None:
    for key, value in attribute_values.items():
        if "start" in key.lower() and isinstance(value, pd.Timestamp):
            return value
    date_values = [value for value in attribute_values.values() if isinstance(value, pd.Timestamp)]
    if date_values:
        return date_values[0]
    return None


def _build_tsf_series_id(attribute_values: dict[str, object], series_idx: int) -> str:
    preferred_keys = ["id", "series_name", "series_id"]
    lowered = {key.lower(): key for key in attribute_values}
    for key in preferred_keys:
        if key in lowered:
            return str(attribute_values[lowered[key]])

    pieces = [str(value) for value in attribute_values.values() if not isinstance(value, pd.Timestamp)]
    if pieces:
        return "__".join(pieces)
    return f"series_{series_idx:04d}"


def _build_tsf_timestamp(
    start_timestamp: pd.Timestamp | None,
    step: int,
    frequency: str,
) -> pd.Timestamp:
    base_timestamp = start_timestamp if start_timestamp is not None else pd.Timestamp("1970-01-01")
    offset = _tsf_frequency_to_offset(frequency)
    if offset is None:
        return base_timestamp + pd.to_timedelta(step, unit="D")
    return base_timestamp + step * offset


def _tsf_frequency_to_offset(frequency: str):
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


def load_table(uploaded_file_name: str, raw_bytes: bytes) -> pd.DataFrame:
    suffix = Path(uploaded_file_name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(raw_bytes))
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(BytesIO(raw_bytes))
    if suffix == ".tsf":
        return parse_tsf(raw_bytes)
    raise ValueError("CSV, Parquet 또는 TSF 파일만 업로드할 수 있습니다.")
