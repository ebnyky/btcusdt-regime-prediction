from __future__ import annotations

from torch import nn

from shared import NUM_CLASSES


class SoftLogisticRegression(nn.Module):
    """Multinomial logistic regression trained directly on soft targets."""

    def __init__(self, input_dim: int = 512, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, inputs):
        return self.linear(inputs)


class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_units: tuple[int, ...] = (256, 64),
        dropout: float = 0.30,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_units:
            layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)))
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


def build_embedding_model(name: str, config: dict):
    if name == "soft_logistic_regression":
        return SoftLogisticRegression()
    if name == "embedding_mlp":
        return EmbeddingMLP(
            hidden_units=tuple(int(x) for x in config.get("mlp_hidden_units", [256, 64])),
            dropout=float(config.get("mlp_dropout", 0.30)),
        )
    raise ValueError(f"Unknown embedding model: {name}")
