"""Stable model boundary shared by native CDSR implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Mapping, MutableMapping, Union

import torch
from torch import nn

LossOutput = Union[torch.Tensor, Mapping[str, torch.Tensor]]


class CDSRModel(nn.Module, ABC):
    """Base class for models trained natively by the CDSR framework."""

    required_batch_fields: frozenset[str] = frozenset(
        {"mixed_item_seq", "mixed_domain_seq", "target_item_id", "target_domain_id"}
    )

    @abstractmethod
    def calculate_loss(self, batch: Mapping[str, torch.Tensor]) -> LossOutput:
        """Return a scalar loss, or a mapping containing a scalar ``loss``."""

    @abstractmethod
    def score_candidates(
        self, batch: Mapping[str, torch.Tensor], candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return a ``[batch, candidates]`` score tensor (larger is better)."""

    def validate_batch(self, batch: Mapping[str, torch.Tensor]) -> None:
        missing = self.required_batch_fields.difference(batch)
        if missing:
            raise KeyError(f"Missing model batch fields: {sorted(missing)}")

    @staticmethod
    def total_loss(output: LossOutput) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            loss = output
        else:
            if "loss" not in output:
                raise KeyError("A mapped loss output must contain key 'loss'")
            loss = output["loss"]
        if loss.ndim != 0:
            raise ValueError(f"Training loss must be scalar, got shape {tuple(loss.shape)}")
        return loss

    @staticmethod
    def loss_log(output: LossOutput) -> Dict[str, float]:
        if isinstance(output, torch.Tensor):
            return {"loss": float(output.detach())}
        return {key: float(value.detach()) for key, value in output.items() if value.ndim == 0}


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device | str
) -> MutableMapping[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
