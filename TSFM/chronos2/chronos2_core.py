from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from chronos import Chronos2Pipeline


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_example_frames(
    num_series: int,
    context_length: int,
    prediction_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_rows: list[dict] = []
    future_rows: list[dict] = []
    base_timestamp = pd.Timestamp("2024-01-01")

    for series_idx in range(num_series):
        series_id = f"series_{series_idx:02d}"
        base_level = 80 + 5 * series_idx

        for step in range(context_length + prediction_length):
            timestamp = base_timestamp + pd.Timedelta(hours=step)
            seasonal = 10 * ((step % 24) / 24.0)
            trend = 0.4 * step
            price_index = 100 + 2 * ((step + series_idx) % 7)
            promo = 1 if step % 12 in (0, 1) else 0
            target = base_level + seasonal + trend - 3 * promo + 0.2 * price_index

            row = {
                "id": series_id,
                "timestamp": timestamp,
                "target": round(target, 3),
                "price_index": float(price_index),
                "promo": int(promo),
            }

            if step < context_length:
                context_rows.append(row)
            else:
                future_rows.append(
                    {
                        "id": series_id,
                        "timestamp": timestamp,
                        "price_index": float(price_index),
                        "promo": int(promo),
                    }
                )

    context_df = pd.DataFrame(context_rows)
    future_df = pd.DataFrame(future_rows)
    return context_df, future_df


def load_pipeline(model_id: str, device: str) -> Chronos2Pipeline:
    return Chronos2Pipeline.from_pretrained(model_id, device_map=device)


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
    if quantile_levels is None:
        quantile_levels = [0.1, 0.5, 0.9]

    prepared_context = context_df.copy()
    prepared_context[timestamp_column] = pd.to_datetime(prepared_context[timestamp_column])
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


def load_table(uploaded_file_name: str, raw_bytes: bytes) -> pd.DataFrame:
    suffix = Path(uploaded_file_name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(raw_bytes))
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(BytesIO(raw_bytes))
    raise ValueError("CSV 또는 Parquet 파일만 업로드할 수 있습니다.")
