from __future__ import annotations

import argparse
from pathlib import Path

from chronos2_core import (
    build_example_frames,
    detect_device,
    load_pipeline,
    run_prediction,
)


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

    pipeline = load_pipeline(args.model_id, device)
    pred_df = run_prediction(
        pipeline=pipeline,
        context_df=context_df,
        future_df=future_df,
        prediction_length=args.prediction_length,
        id_column="id",
        timestamp_column="timestamp",
        target_column="target",
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
