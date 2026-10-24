"""Model-specific views over one immutable canonical protocol.

These adapters translate field layout only. They never split data, resample
evaluation negatives, compute metrics, or alter a model's loss/training stages.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, MutableMapping, Sequence

from dataclasses import dataclass


@dataclass(frozen=True)
class _AdapterSpec:
    trainer_strategy: str
    required_features: tuple


BASELINE_SPECS: dict = {}
METHOD_SPECS: dict = {"comm_cdsr_best": _AdapterSpec(
    trainer_strategy="standard",
    required_features=("mixed-sequence", "train-only-soft-communities",
                       "community-profiles", "soft-community-bridges"))}
from cdsr.data.schema import CanonicalCorpus, SequenceExample
from cdsr.protocol.bundle import ProtocolBundle, ProtocolRow, example_id, validate_protocol_against_corpus


def _domain_views(example: SequenceExample, num_domains: int) -> Dict[int, Dict[str, List[int | float]]]:
    result: Dict[int, Dict[str, List[int | float]]] = {}
    for domain in range(num_domains):
        indices = [index for index, value in enumerate(example.mixed_domain_seq) if value == domain]
        result[domain] = {
            "items": [example.mixed_item_seq[index] for index in indices],
            "local_items": [example.mixed_local_item_seq[index] for index in indices],
            "timestamps": [example.mixed_timestamp_seq[index] for index in indices],
        }
    return result


def _base_row(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    return {
        "example_id": example_id(example),
        "split": example.split,
        "user_id": example.user_id,
        "mixed_item_seq": example.mixed_item_seq,
        "mixed_local_item_seq": example.mixed_local_item_seq,
        "mixed_domain_seq": example.mixed_domain_seq,
        "mixed_timestamp_seq": example.mixed_timestamp_seq,
        "source_item_seq": example.source_item_seq,
        "target_item_seq": example.target_item_seq,
        "domain_sequences": _domain_views(example, num_domains),
        "target_item_id": example.target_item_id,
        "target_local_item_id": example.target_local_item_id,
        "target_domain_id": example.target_domain_id,
        "target_timestamp": example.target_timestamp,
    }


def _with_candidates(value: MutableMapping[str, object], protocol_row: ProtocolRow | None) -> None:
    if protocol_row is None:
        return
    value.update(
        {
            "protocol_id": None,  # populated during export
            "candidate_ids": protocol_row.candidate_ids,
            "target_index": protocol_row.target_index,
            "candidate_fingerprint": protocol_row.candidate_fingerprint,
        }
    )


def _aligned_domain(example: SequenceExample, domain: int) -> List[int]:
    return [
        item if item_domain == domain else 0
        for item, item_domain in zip(example.mixed_item_seq, example.mixed_domain_seq)
    ]


def _abxi(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    if num_domains != 2:
        raise ValueError("ABXI protocol adapter requires two domains")
    value = _base_row(example, num_domains)
    value.update({"seq_x": example.mixed_item_seq, "seq_a": _aligned_domain(example, 0), "seq_b": _aligned_domain(example, 1), "gt": example.target_item_id})
    return value


def _amid(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    value = _base_row(example, num_domains)
    value.update({"sequence": example.mixed_item_seq, "domain_id": example.target_domain_id, "label": 1, "observation_label": 1})
    return value


def _autocdsr(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    value = _base_row(example, num_domains)
    value.update({"input_ids": example.mixed_item_seq, "input_domain_ids": example.mixed_domain_seq, "labels": [example.target_item_id]})
    return value


def _c2dsr(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    if num_domains != 2:
        raise ValueError("C2DSR protocol adapter requires two domains")
    value = _base_row(example, num_domains)
    x = _aligned_domain(example, 0)
    y = _aligned_domain(example, 1)
    value.update(
        {
            "seq": example.mixed_item_seq,
            "x_seq": x,
            "y_seq": y,
            "position": list(range(1, len(example.mixed_item_seq) + 1)),
            "x_position": [index + 1 if item else 0 for index, item in enumerate(x)],
            "y_position": [index + 1 if item else 0 for index, item in enumerate(y)],
            "ground": example.target_item_id,
        }
    )
    return value


def _triple_sequence(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    value = _base_row(example, num_domains)
    value.update({"mix_seq": example.mixed_item_seq, "source_seq": example.source_item_seq, "target_seq": example.target_item_seq})
    return value


def _llm_side(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    value = _triple_sequence(example, num_domains)
    value.update({"item_embedding_ids": example.mixed_item_seq, "user_embedding_id": example.user_id})
    return value


def _semacdr(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    value = _triple_sequence(example, num_domains)
    value.update({"graph_user_id": example.user_id, "graph_item_ids": example.mixed_item_seq, "semantic_item_ids": example.mixed_item_seq})
    return value


def _syncrec(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    value = _base_row(example, num_domains)
    value.update({"cdsr_sequence": example.mixed_item_seq, "sdsr_sequences": _domain_views(example, num_domains), "task_domain": example.target_domain_id})
    return value


def _hgtl(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    value = _base_row(example, num_domains)
    value.update({"main_domain": example.target_domain_id, "sequences_by_domain": _domain_views(example, num_domains), "requires_category_artifact": True})
    return value


def _weaverec(example: SequenceExample, num_domains: int) -> Dict[str, object]:
    value = _base_row(example, num_domains)
    value.update(
        {
            "prompt_payload": {
                "history_item_ids": example.mixed_item_seq,
                "history_domain_ids": example.mixed_domain_seq,
                "target_domain_id": example.target_domain_id,
            },
            "requires_item_text": True,
        }
    )
    return value


ROW_BUILDERS: Dict[str, Callable[[SequenceExample, int], Dict[str, object]]] = {
    "abxi": _abxi,
    "amid": _amid,
    "autocdsr": _autocdsr,
    "c2dsr": _c2dsr,
    "cmgr": _llm_side,
    "hgtl": _hgtl,
    "horizonrec": _triple_sequence,
    "llm4cdsr": _llm_side,
    "semacdr": _semacdr,
    "syncrec": _syncrec,
    "tricdr": _triple_sequence,
    "weaverec": _weaverec,
}

METHOD_ROW_BUILDERS: Dict[str, Callable[[SequenceExample, int], Dict[str, object]]] = {
    "comm_cdsr": _triple_sequence,
    "comm_cdsr_best": _triple_sequence,
    "comm_cdsr_v3": _triple_sequence,
}


@dataclass
class AdapterExport:
    model: str
    output_dir: Path
    adapter_id: str
    files: Dict[str, str]


def export_adapter_data(
    model: str,
    corpus: CanonicalCorpus,
    protocol: ProtocolBundle,
    output_dir: str | Path,
    *,
    validate_protocol: bool = True,
) -> AdapterExport:
    normalized = model.lower()
    builders = {**ROW_BUILDERS, **METHOD_ROW_BUILDERS}
    if normalized not in builders:
        raise KeyError(f"No protocol row builder for {model}")
    if validate_protocol:
        validate_protocol_against_corpus(protocol, corpus)
    destination = Path(output_dir) / normalized
    destination.mkdir(parents=True, exist_ok=True)
    builder = builders[normalized]
    raw_user_by_id = {value: key for key, value in corpus.user_to_id.items()}
    raw_item_by_id = {value: key for key, value in corpus.global_item_to_id.items()}

    def stable_example_id(example: SequenceExample) -> str:
        return example_id(
            example,
            raw_user_id=raw_user_by_id[example.user_id],
            raw_target_item_id=raw_item_by_id[example.target_item_id],
        )

    protocol_rows = {
        split: {row.example_id: row for row in values} for split, values in protocol.rows.items()
    }
    files: Dict[str, str] = {}
    content_hash = hashlib.sha256()
    for split in ("train", "valid", "test"):
        filename = f"{split}.jsonl"
        files[split] = filename
        with (destination / filename).open("w", encoding="utf-8") as stream:
            selected_examples = corpus.examples[split]
            if split in protocol_rows:
                selected_examples = [
                    example
                    for example in selected_examples
                    if stable_example_id(example) in protocol_rows[split]
                ]
            for example in selected_examples:
                value = builder(example, corpus.num_domains)
                value["example_id"] = stable_example_id(example)
                value["user_group"] = example.user_group
                protocol_row = protocol_rows.get(split, {}).get(value["example_id"])
                _with_candidates(value, protocol_row)
                if protocol_row is not None:
                    value["protocol_id"] = protocol.protocol_id
                line = json.dumps(value, ensure_ascii=False, sort_keys=True)
                content_hash.update(line.encode("utf-8"))
                stream.write(line + "\n")
    adapter_id = content_hash.hexdigest()
    spec = {**BASELINE_SPECS, **METHOD_SPECS}[normalized]
    manifest = {
        "model": normalized,
        "protocol_id": protocol.protocol_id,
        "adapter_id": adapter_id,
        "files": files,
        "trainer_strategy": spec.trainer_strategy,
        "required_features": list(spec.required_features),
        "model_code_unchanged": normalized in BASELINE_SPECS,
        "split_owned_by_protocol": True,
        "evaluation_owned_by_protocol": True,
    }
    (destination / "adapter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return AdapterExport(normalized, destination, adapter_id, files)
