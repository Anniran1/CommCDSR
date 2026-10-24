"""Canonical preprocessing, temporal splitting and PyTorch batching for CDSR."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from cdsr.data.schema import (
    CanonicalCorpus,
    CanonicalInteraction,
    RawInteraction,
    SequenceExample,
)


def _parse_timestamp(value: object) -> float:
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        normalized = text.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError as error:
            raise ValueError(f"Unsupported timestamp {text!r}") from error


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as stream:
        sample = stream.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def read_interactions(
    path: str | Path,
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    domain_col: str = "domain_id",
    timestamp_col: str = "timestamp",
    delimiter: Optional[str] = None,
) -> List[RawInteraction]:
    """Read a header-based CSV/TSV file without imposing numeric raw IDs."""

    source = Path(path)
    actual_delimiter = delimiter or _detect_delimiter(source)
    interactions: List[RawInteraction] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=actual_delimiter)
        required = {user_col, item_col, domain_col, timestamp_col}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {source}: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            interactions.append(
                RawInteraction(
                    user=str(row[user_col]),
                    item=str(row[item_col]),
                    domain=str(row[domain_col]),
                    timestamp=_parse_timestamp(row[timestamp_col]),
                    row_number=row_number,
                )
            )
    if not interactions:
        raise ValueError(f"No interactions found in {source}")
    return interactions


def read_domain_interactions(
    path: str | Path,
    *,
    domain: str,
    user_col: str = "user_id",
    item_col: str = "item_id",
    timestamp_col: str = "timestamp",
    rating_col: str | None = None,
    min_rating: float | None = None,
    delimiter: Optional[str] = None,
    row_offset: int = 0,
) -> List[RawInteraction]:
    """Read a RecBole atomic file and assign one explicit domain label.

    Atomic headers such as ``user_id:token`` are normalized to ``user_id``.
    Rating filtering is optional and is performed before canonical ID mapping.
    """

    source = Path(path)
    actual_delimiter = delimiter or _detect_delimiter(source)
    interactions: List[RawInteraction] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter=actual_delimiter)
        try:
            typed_header = next(reader)
        except StopIteration as error:
            raise ValueError(f"Empty interaction file: {source}") from error
        header = [column.split(":", 1)[0] for column in typed_header]
        positions = {name: index for index, name in enumerate(header)}
        required = {user_col, item_col, timestamp_col}
        if rating_col:
            required.add(rating_col)
        missing = required.difference(positions)
        if missing:
            raise ValueError(f"Missing columns in {source}: {sorted(missing)}")
        for physical_row, values in enumerate(reader, start=2):
            if not values:
                continue
            if rating_col and min_rating is not None:
                if float(values[positions[rating_col]]) < min_rating:
                    continue
            interactions.append(
                RawInteraction(
                    user=str(values[positions[user_col]]),
                    item=str(values[positions[item_col]]),
                    domain=domain,
                    timestamp=_parse_timestamp(values[positions[timestamp_col]]),
                    row_number=row_offset + physical_row,
                )
            )
    return interactions


def _stable_mapping(values: Iterable[str], start: int) -> Dict[str, int]:
    return {value: index for index, value in enumerate(sorted(set(values)), start=start)}


def _canonicalize(
    interactions: Sequence[RawInteraction],
) -> tuple[
    List[CanonicalInteraction],
    Dict[str, int],
    Dict[str, int],
    Dict[str, int],
    Dict[str, Dict[str, int]],
    Dict[int, List[int]],
]:
    user_to_id = _stable_mapping((row.user for row in interactions), start=1)
    domain_to_id = _stable_mapping((row.domain for row in interactions), start=0)

    domain_items: Dict[str, set[str]] = defaultdict(set)
    for row in interactions:
        domain_items[row.domain].add(row.item)

    local_item_to_id: Dict[str, Dict[str, int]] = {
        domain: _stable_mapping(items, start=1) for domain, items in domain_items.items()
    }
    global_item_to_id: Dict[str, int] = {}
    domain_catalogs: Dict[int, List[int]] = defaultdict(list)
    next_item_id = 1
    for domain in sorted(domain_items):
        domain_id = domain_to_id[domain]
        for item in sorted(domain_items[domain]):
            key = f"{domain}\u241f{item}"
            global_item_to_id[key] = next_item_id
            domain_catalogs[domain_id].append(next_item_id)
            next_item_id += 1

    canonical: List[CanonicalInteraction] = []
    for row in interactions:
        key = f"{row.domain}\u241f{row.item}"
        canonical.append(
            CanonicalInteraction(
                user_id=user_to_id[row.user],
                item_id=global_item_to_id[key],
                local_item_id=local_item_to_id[row.domain][row.item],
                domain_id=domain_to_id[row.domain],
                timestamp=row.timestamp,
                row_number=row.row_number,
            )
        )
    return (
        canonical,
        user_to_id,
        domain_to_id,
        global_item_to_id,
        local_item_to_id,
        dict(domain_catalogs),
    )


def build_canonical_corpus(
    interactions: Sequence[RawInteraction],
    *,
    max_sequence_length: int = 50,
    min_history: int = 1,
    min_domain_interactions: int = 3,
    split_strategy: str = "per-domain-leave-two-out",
    target_domain: str | None = None,
    split_anchor_domains: Sequence[str] | None = None,
) -> CanonicalCorpus:
    """Create prefix-only, chronologically leakage-free next-item examples.

    ``per-domain-leave-two-out`` selects one validation and test target in every
    eligible domain. Training stops before the earliest validation target across
    domains, which prevents a late source-domain training event from leaking the
    future of an earlier target-domain validation event.

    ``global-leave-two-out`` selects only the final two events per user. Every
    example only sees interactions strictly preceding its target.
    """

    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")
    if min_history < 0:
        raise ValueError("min_history cannot be negative")
    if min_domain_interactions < 3:
        raise ValueError("min_domain_interactions must be at least 3")
    if split_strategy not in {"per-domain-leave-two-out", "global-leave-two-out"}:
        raise ValueError("Unsupported split_strategy")

    (
        canonical,
        user_to_id,
        domain_to_id,
        global_item_to_id,
        local_item_to_id,
        domain_catalogs,
    ) = _canonicalize(interactions)
    anchor_domain_ids: set[int] | None = None
    if split_anchor_domains is not None:
        unknown = set(split_anchor_domains).difference(domain_to_id)
        if unknown:
            raise ValueError(f"Unknown split anchor domains: {sorted(unknown)}")
        anchor_domain_ids = {domain_to_id[name] for name in split_anchor_domains}
        if not anchor_domain_ids:
            raise ValueError("split_anchor_domains cannot be empty")

    raw_domains_by_user: Dict[str, set[str]] = defaultdict(set)
    all_raw_domains = {row.domain for row in interactions}
    for row in interactions:
        raw_domains_by_user[row.user].add(row.domain)
    user_groups: Dict[int, str] = {}
    for raw_user, user_id in user_to_id.items():
        domains = raw_domains_by_user[raw_user]
        # For two or more source domains, ``overlap`` means that the user has
        # target interactions and at least one source-domain interaction.  It
        # must not require presence in every source domain: doing so would
        # incorrectly relabel a Books+Movies user as target-only merely because
        # the user has no CD interaction.
        has_target = target_domain is not None and target_domain in domains
        has_source = target_domain is not None and bool(domains - {target_domain})
        if target_domain is not None and has_target and has_source:
            group = "overlap"
        elif target_domain is None:
            group = "overlap" if domains == all_raw_domains else "non_overlap"
        elif has_target:
            group = "target_only"
        else:
            group = "source_only"
        user_groups[user_id] = group

    by_user: Dict[int, List[CanonicalInteraction]] = defaultdict(list)
    for row in canonical:
        by_user[row.user_id].append(row)

    examples: Dict[str, List[SequenceExample]] = {"train": [], "valid": [], "test": []}
    train_sequences: Dict[int, List[CanonicalInteraction]] = {}
    dropped_short_domain = 0
    for user_id, events in by_user.items():
        events.sort(key=lambda event: (event.timestamp, event.row_number))
        positions_by_domain: Dict[int, List[int]] = defaultdict(list)
        for position, event in enumerate(events):
            positions_by_domain[event.domain_id].append(position)

        eligible_positions: Dict[int, str] = {}
        if split_strategy == "global-leave-two-out":
            if len(events) < 3:
                dropped_short_domain += len(events)
                continue
            for position in range(len(events) - 2):
                eligible_positions[position] = "train"
            eligible_positions[len(events) - 2] = "valid"
            eligible_positions[len(events) - 1] = "test"
            train_sequences[user_id] = list(events[:-2])
        else:
            validation_positions: Dict[int, int] = {}
            for domain_id, positions in positions_by_domain.items():
                if len(positions) < min_domain_interactions:
                    dropped_short_domain += len(positions)
                    continue
                validation_positions[domain_id] = positions[-2]
                eligible_positions[positions[-2]] = "valid"
                eligible_positions[positions[-1]] = "test"
            if validation_positions:
                # A single temporal train/evaluation boundary is essential. A
                # per-domain split without this cutoff can put another domain's
                # test event into the history of a later training example.
                anchored = [
                    position
                    for domain_id, position in validation_positions.items()
                    if anchor_domain_ids is None or domain_id in anchor_domain_ids
                ]
                # A CD-only user has no Books/Movies anchor. Retain that user
                # with the ordinary leave-two-out cutoff in the available domain.
                train_cutoff = min(anchored or list(validation_positions.values()))
                for position in range(train_cutoff):
                    # If none of the anchor domains is split-eligible, the
                    # corresponding two-domain corpus has no training row for
                    # this user. CD may still train on its own history, but it
                    # must not pull otherwise-dropped Books/Movies events into
                    # the multi-domain training view.
                    if (
                        anchor_domain_ids is not None
                        and not anchored
                        and events[position].domain_id in anchor_domain_ids
                    ):
                        continue
                    eligible_positions[position] = "train"
                train_sequences[user_id] = [
                    event
                    for event in events[:train_cutoff]
                    if not (
                        anchor_domain_ids is not None
                        and not anchored
                        and event.domain_id in anchor_domain_ids
                    )
                ]

        for position in sorted(eligible_positions):
            target = events[position]
            full_history = events[:position]
            history = full_history[-max_sequence_length:]
            if len(history) < min_history:
                continue
            # Domain-native models such as AMID truncate each domain
            # independently rather than filtering an already-truncated mixed
            # window. The mixed sequence remains governed by the common input
            # budget used by the frozen protocol.
            target_history = [
                event for event in full_history if event.domain_id == target.domain_id
            ][-max_sequence_length:]
            source_history = [
                event for event in full_history if event.domain_id != target.domain_id
            ][-max_sequence_length:]
            example = SequenceExample(
                user_id=user_id,
                mixed_item_seq=[event.item_id for event in history],
                mixed_local_item_seq=[event.local_item_id for event in history],
                mixed_domain_seq=[event.domain_id for event in history],
                mixed_timestamp_seq=[event.timestamp for event in history],
                source_item_seq=[event.item_id for event in source_history],
                target_item_seq=[event.item_id for event in target_history],
                target_item_id=target.item_id,
                target_local_item_id=target.local_item_id,
                target_domain_id=target.domain_id,
                target_timestamp=target.timestamp,
                split=eligible_positions[position],
                user_group=user_groups[user_id],
            )
            examples[example.split].append(example)

    return CanonicalCorpus(
        examples=examples,
        user_to_id=user_to_id,
        domain_to_id=domain_to_id,
        global_item_to_id=global_item_to_id,
        local_item_to_id=local_item_to_id,
        domain_catalogs=domain_catalogs,
        train_sequences=train_sequences,
        user_groups=user_groups,
        metadata={
            "split_strategy": split_strategy,
            "max_sequence_length": max_sequence_length,
            "min_history": min_history,
            "min_domain_interactions": min_domain_interactions,
            "dropped_short_domain_events": dropped_short_domain,
            "target_domain": target_domain,
            "split_anchor_domains": list(split_anchor_domains or []),
            "user_group_counts": {
                group: sum(value == group for value in user_groups.values())
                for group in sorted(set(user_groups.values()))
            },
        },
    )


def save_corpus(corpus: CanonicalCorpus, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {
        "user_to_id": corpus.user_to_id,
        "domain_to_id": corpus.domain_to_id,
        "global_item_to_id": corpus.global_item_to_id,
        "local_item_to_id": corpus.local_item_to_id,
        "domain_catalogs": {str(key): value for key, value in corpus.domain_catalogs.items()},
        "train_sequence_file": "train_sequences.jsonl",
        "user_groups": {str(key): value for key, value in corpus.user_groups.items()},
        "metadata": corpus.metadata,
        "counts": {split: len(values) for split, values in corpus.examples.items()},
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (destination / "train_sequences.jsonl").open("w", encoding="utf-8") as stream:
        for user_id, interactions in sorted(corpus.train_sequences.items()):
            stream.write(
                json.dumps(
                    {
                        "user_id": user_id,
                        "interactions": [asdict(value) for value in interactions],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    for split, values in corpus.examples.items():
        with (destination / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value.to_dict(), ensure_ascii=False) + "\n")


def load_corpus(path: str | Path) -> CanonicalCorpus:
    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    examples: Dict[str, List[SequenceExample]] = {}
    for split in ("train", "valid", "test"):
        examples[split] = []
        with (source / f"{split}.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    examples[split].append(SequenceExample.from_dict(json.loads(line)))
    train_sequences: Dict[int, List[CanonicalInteraction]] = {}
    train_sequence_path = source / metadata.get("train_sequence_file", "train_sequences.jsonl")
    if train_sequence_path.exists():
        with train_sequence_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    train_sequences[int(value["user_id"])] = [
                        CanonicalInteraction(**interaction)
                        for interaction in value["interactions"]
                    ]
    return CanonicalCorpus(
        examples=examples,
        user_to_id={str(key): int(value) for key, value in metadata["user_to_id"].items()},
        domain_to_id={str(key): int(value) for key, value in metadata["domain_to_id"].items()},
        global_item_to_id={
            str(key): int(value) for key, value in metadata["global_item_to_id"].items()
        },
        local_item_to_id={
            str(domain): {str(key): int(value) for key, value in mapping.items()}
            for domain, mapping in metadata["local_item_to_id"].items()
        },
        domain_catalogs={int(key): list(map(int, value)) for key, value in metadata["domain_catalogs"].items()},
        train_sequences=train_sequences,
        user_groups={
            int(key): str(value)
            for key, value in metadata.get("user_groups", {}).items()
        },
        metadata=dict(metadata.get("metadata", {})),
    )


class CanonicalCDSRDataset(Dataset[SequenceExample]):
    def __init__(self, corpus: CanonicalCorpus, split: str):
        if split not in corpus.examples:
            raise ValueError(f"Unknown split {split!r}")
        self.corpus = corpus
        self.split = split
        self.examples = corpus.examples[split]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SequenceExample:
        return self.examples[index]


def _left_pad(values: Sequence[int | float], length: int, pad: int | float) -> List[int | float]:
    trimmed = list(values[-length:])
    return [pad] * (length - len(trimmed)) + trimmed


def collate_cdsr(examples: Sequence[SequenceExample]) -> Dict[str, torch.Tensor]:
    """Collate variable-length examples using left padding.

    Domain sequence values are shifted by one so that zero remains padding.
    ``target_domain_id`` itself stays zero based.
    """

    if not examples:
        raise ValueError("Cannot collate an empty batch")
    mixed_length = max(1, max(len(example.mixed_item_seq) for example in examples))
    source_length = max(1, max(len(example.source_item_seq) for example in examples))
    target_length = max(1, max(len(example.target_item_seq) for example in examples))

    group_to_id = {"overlap": 0, "source_only": 1, "target_only": 2, "non_overlap": 3}
    batch = {
        "user_id": torch.tensor([example.user_id for example in examples], dtype=torch.long),
        "user_group_id": torch.tensor(
            [group_to_id[example.user_group] for example in examples], dtype=torch.long
        ),
        "mixed_item_seq": torch.tensor(
            [_left_pad(example.mixed_item_seq, mixed_length, 0) for example in examples], dtype=torch.long
        ),
        "mixed_local_item_seq": torch.tensor(
            [_left_pad(example.mixed_local_item_seq, mixed_length, 0) for example in examples], dtype=torch.long
        ),
        "mixed_domain_seq": torch.tensor(
            [
                _left_pad([domain + 1 for domain in example.mixed_domain_seq], mixed_length, 0)
                for example in examples
            ],
            dtype=torch.long,
        ),
        "mixed_timestamp_seq": torch.tensor(
            [_left_pad(example.mixed_timestamp_seq, mixed_length, 0.0) for example in examples],
            dtype=torch.float64,
        ),
        "source_item_seq": torch.tensor(
            [_left_pad(example.source_item_seq, source_length, 0) for example in examples], dtype=torch.long
        ),
        "target_item_seq": torch.tensor(
            [_left_pad(example.target_item_seq, target_length, 0) for example in examples], dtype=torch.long
        ),
        "mixed_length": torch.tensor([len(example.mixed_item_seq) for example in examples], dtype=torch.long),
        "source_length": torch.tensor([len(example.source_item_seq) for example in examples], dtype=torch.long),
        "target_length": torch.tensor([len(example.target_item_seq) for example in examples], dtype=torch.long),
        "target_item_id": torch.tensor([example.target_item_id for example in examples], dtype=torch.long),
        "target_local_item_id": torch.tensor(
            [example.target_local_item_id for example in examples], dtype=torch.long
        ),
        "target_domain_id": torch.tensor(
            [example.target_domain_id for example in examples], dtype=torch.long
        ),
        "target_timestamp": torch.tensor(
            [example.target_timestamp for example in examples], dtype=torch.float64
        ),
    }
    return batch
