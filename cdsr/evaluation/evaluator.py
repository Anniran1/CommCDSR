"""Target-domain candidate construction and ranking evaluation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Sequence

import torch

from cdsr.evaluation.metrics import grouped_metrics, metrics_from_ranks
from cdsr.models.base import CDSRModel, move_batch

if TYPE_CHECKING:
    from cdsr.protocol.bundle import ProtocolBundle


@dataclass(frozen=True)
class EvaluationConfig:
    topk: tuple[int, ...] = (5, 10, 20)
    candidate_mode: str = "full"
    num_sampled_negatives: int = 999
    filter_history: bool = True
    seed: int = 2026
    record_per_user_accuracy: bool = False
    per_user_splits: tuple[str, ...] = ("test",)

    def __post_init__(self) -> None:
        if self.candidate_mode not in {"full", "sampled"}:
            raise ValueError("candidate_mode must be 'full' or 'sampled'")
        if not self.topk or min(self.topk) < 1:
            raise ValueError("topk must contain positive integers")


class CDSREvaluator:
    def __init__(
        self,
        domain_catalogs: Mapping[int, Sequence[int]],
        config: EvaluationConfig | None = None,
        protocol_bundle: "ProtocolBundle | None" = None,
    ) -> None:
        self.domain_catalogs = {int(key): list(map(int, values)) for key, values in domain_catalogs.items()}
        self.config = config or EvaluationConfig()
        self.protocol_bundle = protocol_bundle

    def _candidates(
        self, domain: int, target: int, history: set[int], row_seed: int
    ) -> List[int]:
        values = [
            item
            for item in self.domain_catalogs[domain]
            if item == target or not (self.config.filter_history and item in history)
        ]
        negatives = [item for item in values if item != target]
        if self.config.candidate_mode == "sampled":
            rng = random.Random(self.config.seed + row_seed)
            sample_size = min(self.config.num_sampled_negatives, len(negatives))
            negatives = rng.sample(negatives, sample_size)
        return [target, *negatives]

    @staticmethod
    def _rank(scores: torch.Tensor, target_index: int = 0) -> int:
        # This tie policy is stable and pessimistic: equal-scored negatives are
        # ranked ahead of the target.
        positive = scores[target_index]
        indices = torch.arange(scores.shape[0], device=scores.device) != target_index
        return int(torch.sum(scores[indices] >= positive).item()) + 1

    @torch.no_grad()
    def evaluate(
        self,
        model: CDSRModel,
        dataloader: Iterable[Mapping[str, torch.Tensor]],
        *,
        device: torch.device | str,
        split: str | None = None,
    ) -> Dict[str, object]:
        model.eval()
        all_ranks: List[int] = []
        ranks_by_domain: Dict[str, List[int]] = defaultdict(list)
        ranks_by_overlap: Dict[str, List[int]] = defaultdict(list)
        ranks_by_user_group: Dict[str, List[int]] = defaultdict(list)
        per_user_rows: Dict[int, List[dict[str, int | str]]] = defaultdict(list)
        record_users = self.config.record_per_user_accuracy and split in self.config.per_user_splits
        row_seed = 0
        protocol_rows = None
        if self.protocol_bundle is not None:
            if split not in self.protocol_bundle.rows:
                raise ValueError("A valid protocol split is required for frozen-candidate evaluation")
            protocol_rows = self.protocol_bundle.rows[split]
        for raw_batch in dataloader:
            batch = move_batch(raw_batch, device)
            model.validate_batch(batch)
            candidate_rows: List[List[int]] = []
            target_indices: List[int] = []
            for row in range(batch["target_item_id"].shape[0]):
                target = int(batch["target_item_id"][row])
                domain = int(batch["target_domain_id"][row])
                if protocol_rows is not None:
                    if row_seed >= len(protocol_rows):
                        raise ValueError("Dataloader contains more rows than the frozen protocol")
                    protocol_row = protocol_rows[row_seed]
                    if (
                        protocol_row.user_id != int(batch["user_id"][row])
                        or protocol_row.target_item_id != target
                        or protocol_row.target_domain_id != domain
                    ):
                        raise ValueError("Dataloader row order does not match the frozen protocol")
                    group_ids = {"overlap": 0, "source_only": 1, "target_only": 2}
                    if group_ids[protocol_row.user_group] != int(batch["user_group_id"][row]):
                        raise ValueError("Dataloader user group does not match the frozen protocol")
                    candidate_rows.append(protocol_row.candidate_ids)
                    target_indices.append(protocol_row.target_index)
                else:
                    history = set(map(int, batch["mixed_item_seq"][row].tolist()))
                    history.discard(0)
                    candidate_rows.append(self._candidates(domain, target, history, row_seed))
                    target_indices.append(0)
                row_seed += 1

            max_candidates = max(map(len, candidate_rows))
            candidate_tensor = torch.zeros(
                (len(candidate_rows), max_candidates), dtype=torch.long, device=device
            )
            valid_mask = torch.zeros_like(candidate_tensor, dtype=torch.bool)
            for row, candidates in enumerate(candidate_rows):
                candidate_tensor[row, : len(candidates)] = torch.tensor(candidates, device=device)
                valid_mask[row, : len(candidates)] = True

            scores = model.score_candidates(batch, candidate_tensor)
            if scores.shape != candidate_tensor.shape:
                raise ValueError(
                    f"score_candidates returned {tuple(scores.shape)}, expected {tuple(candidate_tensor.shape)}"
                )
            scores = scores.masked_fill(~valid_mask, float("-inf"))
            for row, candidates in enumerate(candidate_rows):
                rank = self._rank(scores[row, : len(candidates)], target_indices[row])
                domain = int(batch["target_domain_id"][row])
                group_id = int(batch["user_group_id"][row])
                user_group = {0: "overlap", 1: "source_only", 2: "target_only", 3: "non_overlap"}[
                    group_id
                ]
                all_ranks.append(rank)
                ranks_by_domain[f"domain_{domain}"].append(rank)
                ranks_by_overlap[
                    "overlap" if user_group == "overlap" else "non_overlap"
                ].append(rank)
                ranks_by_user_group[user_group].append(rank)
                if record_users:
                    per_user_rows[int(batch["user_id"][row])].append(
                        {
                            "rank": rank,
                            "candidate_count": len(candidates),
                            "target_domain_id": domain,
                            "user_group": user_group,
                        }
                    )

        if protocol_rows is not None and row_seed != len(protocol_rows):
            raise ValueError("Dataloader contains fewer rows than the frozen protocol")
        protocol_result = {
            "candidate_mode": self.config.candidate_mode,
            "filter_history": self.config.filter_history,
            "num_sampled_negatives": self.config.num_sampled_negatives,
        }
        if self.protocol_bundle is not None:
            protocol_result["protocol_id"] = self.protocol_bundle.protocol_id
            protocol_result["frozen_candidates"] = True
        overlap_result = metrics_from_ranks(ranks_by_overlap.get("overlap", []), self.config.topk)
        non_overlap_result = metrics_from_ranks(
            ranks_by_overlap.get("non_overlap", []), self.config.topk
        )
        result = {
            "overall": metrics_from_ranks(all_ranks, self.config.topk),
            "overlap": overlap_result,
            "non_overlap": non_overlap_result,
            "by_domain": grouped_metrics(ranks_by_domain, self.config.topk),
            "by_user_group": grouped_metrics(ranks_by_user_group, self.config.topk),
            "protocol": protocol_result,
        }
        if record_users:
            per_user = []
            for user_id, rows in sorted(per_user_rows.items()):
                ranks = [int(row["rank"]) for row in rows]
                record: dict[str, object] = {
                    "user_id": user_id,
                    "user_group": rows[0]["user_group"],
                    "target_domain_id": rows[0]["target_domain_id"],
                    "examples": len(rows),
                    "rank": sum(ranks) / len(ranks),
                    "candidate_count": sum(int(row["candidate_count"]) for row in rows) / len(rows),
                    # In single-positive ranking, top-1 classification accuracy
                    # is exactly the indicator that the positive ranks first.
                    "accuracy": sum(rank == 1 for rank in ranks) / len(ranks),
                    "reciprocal_rank": sum(1.0 / rank for rank in ranks) / len(ranks),
                }
                for k in sorted(set(self.config.topk)):
                    record[f"accuracy@{k}"] = sum(rank <= k for rank in ranks) / len(ranks)
                    record[f"ndcg@{k}"] = sum(
                        (1.0 / math.log2(rank + 1)) if rank <= k else 0.0
                        for rank in ranks
                    ) / len(ranks)
                per_user.append(record)
            result["per_user"] = per_user
            result["per_user_accuracy_definition"] = (
                "accuracy=I(rank=1); accuracy@K=I(rank<=K), averaged if a user has multiple rows"
            )
        return result
