"""Single-positive ranking metrics used consistently by every adapter."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Mapping


def metrics_from_ranks(ranks: Iterable[int], topk: Iterable[int]) -> Dict[str, float]:
    rank_values = list(ranks)
    if not rank_values:
        return {}
    result: Dict[str, float] = {}
    for k in sorted(set(topk)):
        hits = [rank <= k for rank in rank_values]
        hit = sum(hits) / len(rank_values)
        result[f"hit@{k}"] = hit
        result[f"recall@{k}"] = hit
        result[f"ndcg@{k}"] = sum(
            (1.0 / math.log2(rank + 1)) if rank <= k else 0.0 for rank in rank_values
        ) / len(rank_values)
        result[f"mrr@{k}"] = sum(
            (1.0 / rank) if rank <= k else 0.0 for rank in rank_values
        ) / len(rank_values)
    result["count"] = float(len(rank_values))
    return result


def grouped_metrics(
    ranks_by_group: Mapping[str, Iterable[int]], topk: Iterable[int]
) -> Dict[str, Dict[str, float]]:
    return {group: metrics_from_ranks(ranks, topk) for group, ranks in ranks_by_group.items()}
