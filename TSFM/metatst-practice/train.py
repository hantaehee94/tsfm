from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import SyntheticMetaTSTDataset
from src.model import MetaForecastTransformer


@dataclass
class TrainConfig:
    """Experiment settings for the starter training loop."""

    context_length: int = 48
    prediction_length: int = 24
    num_series: int = 256
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 7


def set_seed(seed: int) -> None:
    """Keep runs reproducible enough for debugging and basic comparisons."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(config: TrainConfig) -> dict[str, DataLoader]:
    """Build train/validation/test loaders from the same synthetic recipe."""

    common = {
        "num_series": config.num_series,
        "context_length": config.context_length,
        "prediction_length": config.prediction_length,
        "seed": config.seed,
    }
    train_dataset = SyntheticMetaTSTDataset(split="train", **common)
    val_dataset = SyntheticMetaTSTDataset(split="val", **common)
    test_dataset = SyntheticMetaTSTDataset(split="test", **common)

    return {
        "train": DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False),
    }


def run_epoch(
    model: MetaForecastTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one epoch for either training or evaluation.

    If `optimizer` is provided, gradients are enabled and parameters are
    updated. Otherwise this function behaves like an evaluation loop.
    """

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    for batch in loader:
        past_values = batch["past_values"].to(device)
        future_values = batch["future_values"].to(device)
        metadata_categorical = batch["metadata_categorical"].to(device)
        metadata_real = batch["metadata_real"].to(device)

        with torch.set_grad_enabled(is_train):
            # The model predicts the full future horizon in one forward pass.
            predictions = model(past_values, metadata_categorical, metadata_real)
            loss = criterion(predictions, future_values)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * past_values.size(0)

    return total_loss / len(loader.dataset)


def save_checkpoint(
    model: MetaForecastTransformer,
    config: TrainConfig,
    save_dir: Path,
) -> Path:
    """Store the best-performing checkpoint for later inspection."""

    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "metatst_transformer.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config)}, checkpoint_path)
    return checkpoint_path


def main() -> None:
    # Keep the CLI intentionally small for the first round of experiments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-series", type=int, default=256)
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_series=args.num_series,
    )
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset metadata schema is reused to configure the forecaster.
    loaders = build_loaders(config)
    model = MetaForecastTransformer(
        context_length=config.context_length,
        prediction_length=config.prediction_length,
        categorical_cardinalities=SyntheticMetaTSTDataset.metadata_cardinalities(),
        num_real_features=SyntheticMetaTSTDataset.metadata_dim(),
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss = run_epoch(model, loaders["val"], criterion, device)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            checkpoint_path = save_checkpoint(
                model=model,
                config=config,
                save_dir=Path("artifacts"),
            )

    # Final test loss is reported after the training loop for a quick sanity check.
    test_loss = run_epoch(model, loaders["test"], criterion, device)
    print(f"test_loss={test_loss:.4f}")
    print(f"best_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
