"""Strict prediction interchange and centralized offline evaluation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping

from cdsr.evaluation.metrics import grouped_metrics, metrics_from_ranks
from cdsr.protocol.bundle import ProtocolBundle, ProtocolRow


def write_prediction_template(
    bundle: ProtocolBundle, split: str, output_file: str | Path
) -> None:
    with Path(output_file).open("w", encoding="utf-8") as stream:
        for row in bundle.rows[split]:
            value = {
                "protocol_id": bundle.protocol_id,
                "example_id": row.example_id,
                "candidate_fingerprint": row.candidate_fingerprint,
                "candidate_ids": row.candidate_ids,
                "scores": None,
            }
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _rank_from_prediction(row: ProtocolRow, prediction: Mapping[str, object]) -> int:
    if prediction.get("scores") is not None:
        scores = list(map(float, prediction["scores"]))
        if len(scores) != len(row.candidate_ids):
            raise ValueError(f"Score count mismatch for {row.example_id}")
        target_score = scores[row.target_index]
        return 1 + sum(
            score >= target_score for index, score in enumerate(scores) if index != row.target_index
        )
    ranked = list(map(int, prediction.get("ranked_item_ids", [])))
    if len(ranked) != len(set(ranked)):
        raise ValueError(f"Duplicate ranked items for {row.example_id}")
    if not set(ranked).issubset(set(row.candidate_ids)):
        raise ValueError(f"Prediction contains an out-of-protocol item for {row.example_id}")
    try:
        return ranked.index(row.target_item_id) + 1
    except ValueError:
        return len(row.candidate_ids) + 1


def evaluate_prediction_file(
    bundle: ProtocolBundle, split: str, prediction_file: str | Path
) -> Dict[str, object]:
    expected = {row.example_id: row for row in bundle.rows[split]}
    predictions: Dict[str, Mapping[str, object]] = {}
    with Path(prediction_file).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("protocol_id") != bundle.protocol_id:
                raise ValueError("Prediction protocol_id mismatch")
            row_id = str(value["example_id"])
            if row_id in predictions:
                raise ValueError(f"Duplicate prediction {row_id}")
            predictions[row_id] = value
    if set(predictions) != set(expected):
        missing = set(expected) - set(predictions)
        extra = set(predictions) - set(expected)
        raise ValueError(f"Prediction coverage mismatch; missing={len(missing)}, extra={len(extra)}")

    ranks: List[int] = []
    by_domain: Dict[str, List[int]] = defaultdict(list)
    for row_id, row in expected.items():
        prediction = predictions[row_id]
        if prediction.get("candidate_fingerprint") != row.candidate_fingerprint:
            raise ValueError(f"Candidate fingerprint mismatch for {row_id}")
        rank = _rank_from_prediction(row, prediction)
        ranks.append(rank)
        by_domain[f"domain_{row.target_domain_id}"].append(rank)
    topk = tuple(map(int, bundle.manifest["config"]["topk"]))
    return {
        "protocol_id": bundle.protocol_id,
        "split": split,
        "overall": metrics_from_ranks(ranks, topk),
        "by_domain": grouped_metrics(by_domain, topk),
    }
