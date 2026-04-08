from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SampleMetadata:
    """Human-readable metadata schema kept for future real-data extensions."""

    series_id: str
    region: str
    category: str
    scale: float


class SyntheticMetaTSTDataset(Dataset):
    """Synthetic dataset for quick metadata-aware forecasting experiments.

    Each sample contains:
    - `past_values`: encoder input window
    - `future_values`: prediction target window
    - `metadata_categorical`: category ids such as region / domain
    - `metadata_real`: continuous descriptors such as scale
    """

    def __init__(
        self,
        num_series: int = 256,
        context_length: int = 48,
        prediction_length: int = 24,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.split = split

        # We generate one full synthetic corpus first, then slice it into
        # train/validation/test splits so each split shares the same recipe.
        records = self._build_records(
            num_series=num_series,
            context_length=context_length,
            prediction_length=prediction_length,
            seed=seed,
        )
        self.records = self._split_records(records, split, train_ratio, val_ratio)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        return {
            # Shape: [context_length]
            "past_values": torch.tensor(record["past_values"], dtype=torch.float32),
            # Shape: [prediction_length]
            "future_values": torch.tensor(record["future_values"], dtype=torch.float32),
            # Shape: [num_categorical_features]
            "metadata_categorical": torch.tensor(
                record["metadata_categorical"], dtype=torch.long
            ),
            # Shape: [num_real_features]
            "metadata_real": torch.tensor(record["metadata_real"], dtype=torch.float32),
        }

    @staticmethod
    def metadata_cardinalities() -> List[int]:
        # region has 4 categories, category has 3 categories.
        return [4, 3]

    @staticmethod
    def metadata_dim() -> int:
        # We currently use one continuous feature: scale.
        return 1

    @staticmethod
    def to_dataframe(records: List[Dict[str, np.ndarray]]) -> pd.DataFrame:
        rows = []
        for record in records:
            rows.append(
                {
                    "series_id": record["series_id"],
                    "region": record["region"],
                    "category": record["category"],
                    "scale": record["metadata_real"][0],
                    "past_mean": float(record["past_values"].mean()),
                    "future_mean": float(record["future_values"].mean()),
                }
            )
        return pd.DataFrame(rows)

    def preview_frame(self) -> pd.DataFrame:
        return self.to_dataframe(self.records[: min(10, len(self.records))])

    def _build_records(
        self,
        num_series: int,
        context_length: int,
        prediction_length: int,
        seed: int,
    ) -> List[Dict[str, np.ndarray]]:
        rng = np.random.default_rng(seed)
        regions = ["north", "south", "east", "west"]
        categories = ["retail", "energy", "traffic"]
        region_to_idx = {value: idx for idx, value in enumerate(regions)}
        category_to_idx = {value: idx for idx, value in enumerate(categories)}
        total_length = context_length + prediction_length
        time_index = np.arange(total_length, dtype=np.float32)

        records: List[Dict[str, np.ndarray]] = []
        for series_idx in range(num_series):
            # Metadata determines how each series is generated. This makes the
            # metadata genuinely informative instead of just extra labels.
            region = regions[series_idx % len(regions)]
            category = categories[(series_idx // len(regions)) % len(categories)]
            scale = 0.8 + 0.15 * (series_idx % 7)

            # Region controls amplitude, while category controls frequency.
            amplitude = {
                "north": 1.0,
                "south": 1.4,
                "east": 0.7,
                "west": 1.8,
            }[region]
            frequency = {
                "retail": 0.18,
                "energy": 0.25,
                "traffic": 0.11,
            }[category]
            trend = 0.015 * (series_idx % 5)
            phase = 0.3 * (series_idx % 9)

            # The final series mixes periodic signal, seasonality, trend, and noise.
            signal = amplitude * np.sin(frequency * time_index + phase)
            seasonal = 0.5 * np.cos(0.05 * time_index + phase / 2.0)
            baseline = trend * time_index + scale
            noise = rng.normal(loc=0.0, scale=0.08 * scale, size=total_length)
            values = signal + seasonal + baseline + noise

            records.append(
                {
                    "series_id": f"series_{series_idx:04d}",
                    "region": region,
                    "category": category,
                    "metadata_categorical": np.array(
                        [region_to_idx[region], category_to_idx[category]], dtype=np.int64
                    ),
                    "metadata_real": np.array([scale], dtype=np.float32),
                    "past_values": values[:context_length].astype(np.float32),
                    "future_values": values[context_length:].astype(np.float32),
                }
            )
        return records

    @staticmethod
    def _split_records(
        records: List[Dict[str, np.ndarray]],
        split: str,
        train_ratio: float,
        val_ratio: float,
    ) -> List[Dict[str, np.ndarray]]:
        total = len(records)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        # Series-wise splitting is enough for this starter project because each
        # record already contains a full context/target pair.
        if split == "train":
            return records[:train_end]
        if split == "val":
            return records[train_end:val_end]
        if split == "test":
            return records[val_end:]
        raise ValueError(f"Unsupported split: {split}")
