"""Default one-optimizer trainer used by straightforward native ports."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import torch

from cdsr.evaluation import CDSREvaluator
from cdsr.models.base import CDSRModel, move_batch


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    valid_metric: str = "ndcg@10"
    patience: int = 10
    # Optional StepLR decay; the defaults keep a constant learning rate.
    lr_decay_step: int = 1
    lr_decay_rate: float = 1.0
    # If > 1, average the top-k validation-scoring epoch snapshots (SWA)
    # and adopt the result when it beats the single best epoch on valid.
    swa_topk: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.swa_topk < 0:
            raise ValueError("swa_topk must be non-negative")
        if self.lr_decay_step < 1:
            raise ValueError("lr_decay_step must be at least 1")
        if self.lr_decay_rate <= 0.0:
            raise ValueError("lr_decay_rate must be positive")


class StandardTrainer:
    def __init__(
        self,
        model: CDSRModel,
        config: TrainerConfig | None = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.logger = logger
        self.optimizer = optimizer or torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(self.config.lr_decay_step),
            gamma=float(self.config.lr_decay_rate),
        )

    def train_epoch(self, dataloader: Iterable[Mapping[str, torch.Tensor]]) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = defaultdict(float)
        steps = 0
        for raw_batch in dataloader:
            batch = move_batch(raw_batch, self.device)
            self.model.validate_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model.calculate_loss(batch)
            loss = self.model.total_loss(output)
            loss.backward()
            if self.config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            for key, value in self.model.loss_log(output).items():
                totals[key] += value
            steps += 1
        return {key: value / max(steps, 1) for key, value in totals.items()}

    def fit(
        self,
        train_loader: Iterable[Mapping[str, torch.Tensor]],
        *,
        valid_loader: Optional[Iterable[Mapping[str, torch.Tensor]]] = None,
        evaluator: Optional[CDSREvaluator] = None,
    ) -> Dict[str, object]:
        history = []
        best_epoch: int | None = None
        best_valid_score = float("-inf")
        best_state: dict[str, torch.Tensor] | None = None
        snapshots: list[tuple[float, int, dict[str, torch.Tensor]]] = []
        stale_epochs = 0
        stopped_early = False
        for epoch in range(1, self.config.epochs + 1):
            if hasattr(self.model, "set_training_epoch"):
                self.model.set_training_epoch(epoch - 1)
            train_start = time.perf_counter()
            train_result = self.train_epoch(train_loader)
            train_seconds = time.perf_counter() - train_start
            record: Dict[str, object] = {
                "epoch": epoch,
                "train": train_result,
                "training_seconds": train_seconds,
            }
            if self.logger is not None:
                losses = ", ".join(f"{key}: {value:.4f}" for key, value in train_result.items())
                self.logger.info(
                    "epoch %d training [time: %.2fs, %s]", epoch - 1, train_seconds, losses
                )
            if valid_loader is not None and evaluator is not None:
                eval_start = time.perf_counter()
                valid_result = evaluator.evaluate(
                    self.model, valid_loader, device=self.device, split="valid"
                )
                eval_seconds = time.perf_counter() - eval_start
                record["valid"] = valid_result
                metrics = valid_result["overall"]
                valid_score = float(metrics.get(self.config.valid_metric, float("nan")))
                if math.isfinite(valid_score) and valid_score > best_valid_score:
                    best_epoch = epoch - 1
                    best_valid_score = valid_score
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if math.isfinite(valid_score):
                    snapshots.append((valid_score, epoch - 1, {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }))
                if self.logger is not None:
                    self.logger.info(
                        "epoch %d evaluating [time: %.2fs, valid_score: %.6f]",
                        epoch - 1,
                        eval_seconds,
                        valid_score,
                    )
                    metric_text = "    ".join(
                        f"{key} : {value:.4f}"
                        for key, value in metrics.items()
                        if key != "count"
                    )
                    self.logger.info("valid result (overall):\n%s", metric_text)
                    for group in ("overlap", "non_overlap"):
                        group_metrics = valid_result.get(group, {})
                        group_text = "    ".join(
                            f"{key} : {value:.4f}"
                            for key, value in group_metrics.items()
                            if key != "count"
                        )
                        self.logger.info(
                            "valid result (%s, count=%d):\n%s",
                            group,
                            int(group_metrics.get("count", 0)),
                            group_text or "<empty>",
                        )
            history.append(record)
            self.scheduler.step()
            if valid_loader is not None and evaluator is not None and stale_epochs >= self.config.patience:
                stopped_early = True
                if self.logger is not None:
                    self.logger.info(
                        "Early stopping triggered at epoch %d after %d epochs without improvement",
                        epoch - 1,
                        self.config.patience,
                    )
                break

        swa: Dict[str, object] | None = None
        if self.config.swa_topk > 1 and len(snapshots) >= self.config.swa_topk:
            ranked = sorted(snapshots, key=lambda item: item[0], reverse=True)
            top = ranked[: self.config.swa_topk]
            reference = top[0][2]
            averaged: dict[str, torch.Tensor] = {}
            for key, value in reference.items():
                if value.dtype.is_floating_point:
                    averaged[key] = torch.stack(
                        [snapshot[2][key].float() for snapshot in top]
                    ).mean(dim=0).to(value.dtype)
                else:
                    averaged[key] = value.clone()
            self.model.load_state_dict(averaged)
            swa_result = evaluator.evaluate(
                self.model, valid_loader, device=self.device, split="valid"
            )
            swa_score = float(
                swa_result["overall"].get(self.config.valid_metric, float("nan"))
            )
            adopted = bool(math.isfinite(swa_score) and swa_score > best_valid_score)
            swa = {
                "swa_epochs": [snapshot[1] for snapshot in top],
                "swa_epoch_scores": [snapshot[0] for snapshot in top],
                "swa_valid_score": swa_score,
                "adopted": adopted,
            }
            if self.logger is not None:
                self.logger.info(
                    "SWA top-%d over epochs %s (scores %s): valid %.6f vs best %.6f -> %s",
                    self.config.swa_topk,
                    swa["swa_epochs"],
                    [round(score, 6) for score in swa["swa_epoch_scores"]],
                    swa_score,
                    best_valid_score,
                    "adopted" if adopted else "rejected",
                )
            if adopted:
                best_state = averaged
                best_valid_score = swa_score
                for record in history:
                    if int(record.get("epoch", -1)) - 1 == int(best_epoch):
                        record["valid"] = swa_result
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if self.logger is not None and best_epoch is not None:
            self.logger.info(
                "best_epoch = %d, best_valid_score = %.6f",
                best_epoch,
                best_valid_score,
            )
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_valid_score": best_valid_score if best_epoch is not None else None,
            "epochs_trained": len(history),
            "stopped_early": stopped_early,
            "swa": swa,
        }
