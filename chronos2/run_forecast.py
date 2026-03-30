from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a minimal local Chronos-2 forecasting experiment."
    )
    parser.add_argument(
        "--model-id",
        default="amazon/chronos-2",
        help="Hugging Face model id to load.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=96,
        help="Number of historical steps per series.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=24,
        help="Number of future steps to forecast.",
    )
    parser.add_argument(
        "--num-series",
        type=int,
        default=3,
        help="Number of related time series in the synthetic example.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions.parquet"),
        help="Path to save the forecast dataframe.",
    )
    args = parser.parse_args()

    device = detect_device()
    print(f"Loading {args.model_id} on device={device}")

    context_df, future_df = build_example_frames(
        num_series=args.num_series,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )

    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map=device)
    pred_df = pipeline.predict_df(
        context_df=context_df,
        future_df=future_df,
        prediction_length=args.prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(args.output, index=False)

    print("\nContext sample:")
    print(context_df.head())
    print("\nForecast sample:")
    print(pred_df.head())
    print(f"\nSaved forecast to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
