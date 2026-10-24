"""Baseline-native training views derived from the shared canonical split."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from cdsr.data.schema import CanonicalCorpus


def _left_pad(values: Sequence[int], length: int, pad: int = 0) -> List[int]:
    values = list(values[-length:])
    return [pad] * (length - len(values)) + values


@dataclass(frozen=True)
class ABXITrainingRecord:
    user_id: int
    sequence: List[int]
    domains: List[int]
    full_seen_items: List[int]
    sequence_length: int


@dataclass(frozen=True)
class CommCDSRTrainingRecord:
    """One target-domain causal sequence for community-transfer training."""

    user_id: int
    target_sequence: List[int]
    full_seen_items: List[int]
    sequence_length: int


class CommCDSRTrainingDataset(Dataset[CommCDSRTrainingRecord]):
    """Expose every target-domain transition exactly once per user.

    Cross-domain knowledge is supplied exclusively by identity-free community
    memory, so no source-user or mixed-user sequence enters this view.
    """

    def __init__(self, corpus: CanonicalCorpus, sequence_length: int,
                 target_domain_id: int):
        if sequence_length < 1:
            raise ValueError("CommCDSR sequence_length must be positive")
        self.records: List[CommCDSRTrainingRecord] = []
        for user_id, events in sorted(corpus.train_sequences.items()):
            full = [event.item_id for event in events
                    if event.domain_id == target_domain_id]
            if len(full) < 2:
                continue
            self.records.append(CommCDSRTrainingRecord(
                user_id=user_id,
                target_sequence=full[-(sequence_length + 1):],
                full_seen_items=full,
                sequence_length=sequence_length,
            ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CommCDSRTrainingRecord:
        return self.records[index]


class CommCDSRCollator:
    """Build target-only all-position labels and unseen multi-negatives."""

    def __init__(self, corpus: CanonicalCorpus, target_domain_id: int,
                 num_negatives: int, seed: int):
        if num_negatives < 1:
            raise ValueError("CommCDSR num_negatives must be positive")
        self.catalog = tuple(corpus.domain_catalogs[target_domain_id])
        self.num_negatives = num_negatives
        self.rng = random.Random(seed)

    def __call__(self, records: Sequence[CommCDSRTrainingRecord]) -> Dict[str, torch.Tensor]:
        if not records:
            raise ValueError("Cannot collate an empty CommCDSR batch")
        length = records[0].sequence_length
        if any(record.sequence_length != length for record in records):
            raise ValueError("CommCDSR batch mixes incompatible training views")

        contexts = []
        positives, negatives, negative_masks = [], [], []
        context_length = length + 1
        for record in records:
            sequence = record.target_sequence
            contexts.append(_left_pad(sequence, context_length))
            row_positives = _left_pad(sequence[1:], length)
            positives.append(row_positives)
            available = [item for item in self.catalog
                         if item not in set(record.full_seen_items)]
            rows, masks = [], []
            for positive in row_positives:
                sampled = (
                    self.rng.sample(available, min(self.num_negatives, len(available)))
                    if positive and available else []
                )
                rows.append(sampled + [0] * (self.num_negatives - len(sampled)))
                masks.append([True] * len(sampled)
                             + [False] * (self.num_negatives - len(sampled)))
            negatives.append(rows)
            negative_masks.append(masks)
        return {
            "comm_user_id": torch.tensor([record.user_id for record in records], dtype=torch.long),
            "comm_context": torch.tensor(contexts, dtype=torch.long),
            "comm_positive": torch.tensor(positives, dtype=torch.long),
            "comm_negative": torch.tensor(negatives, dtype=torch.long),
            "comm_negative_mask": torch.tensor(negative_masks, dtype=torch.bool),
        }


class ABXITrainingDataset(Dataset[ABXITrainingRecord]):
    """One full causal sequence per overlapping user, as in official ABXI."""

    def __init__(self, corpus: CanonicalCorpus, sequence_length: int):
        if sequence_length < 1:
            raise ValueError("ABXI sequence_length must be positive")
        self.sequence_length = sequence_length
        self.records: List[ABXITrainingRecord] = []
        for user_id, events in sorted(corpus.train_sequences.items()):
            if corpus.user_groups.get(user_id) != "overlap" or len(events) < 2:
                continue
            trimmed = events[-(sequence_length + 1) :]
            self.records.append(
                ABXITrainingRecord(
                    user_id=user_id,
                    sequence=[event.item_id for event in trimmed],
                    domains=[event.domain_id for event in trimmed],
                    full_seen_items=[event.item_id for event in events],
                    sequence_length=sequence_length,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ABXITrainingRecord:
        return self.records[index]


def _abxi_domain_sequences(
    sequence: Sequence[int], domains: Sequence[int]
) -> tuple[List[int], List[int], List[int], List[int]]:
    """Torch-independent equivalent of official ``process_train/get_spe_seq``."""

    seq_x = list(sequence[:-1])
    gt = list(sequence[1:])
    domain_x = list(domains[:-1])
    domain_gt = list(domains[1:])
    items_a = [item for item, domain in zip(seq_x, domain_x) if domain == 0]
    items_b = [item for item, domain in zip(seq_x, domain_x) if domain == 1]
    slots_a = [index for index, domain in enumerate(domain_gt) if domain == 0]
    slots_b = [index for index, domain in enumerate(domain_gt) if domain == 1]

    # Official task-guided alignment removes the first cross-domain target and
    # the last source item before assigning domain-specific sequences.
    if domain_x[0] == 1 and slots_a:
        slots_a.pop(0)
    elif domain_x[0] == 0 and slots_b:
        slots_b.pop(0)
    if domain_gt[-1] == 1 and items_a:
        items_a.pop()
    elif domain_gt[-1] == 0 and items_b:
        items_b.pop()

    seq_a = [0] * len(seq_x)
    seq_b = [0] * len(seq_x)
    if len(items_a) != len(slots_a) or len(items_b) != len(slots_b):
        raise ValueError("ABXI task-guided domain alignment is inconsistent")
    for slot, item in zip(slots_a, items_a):
        seq_a[slot] = item
    for slot, item in zip(slots_b, items_b):
        seq_b[slot] = item
    return seq_x, seq_a, seq_b, gt


def collate_abxi_training(records: Sequence[ABXITrainingRecord]) -> Dict[str, torch.Tensor]:
    if not records:
        raise ValueError("Cannot collate an empty ABXI batch")
    length = records[0].sequence_length
    if any(record.sequence_length != length for record in records):
        raise ValueError("ABXI batch mixes incompatible sequence lengths")
    seen_length = max(len(record.full_seen_items) for record in records)
    seq_x, seq_a, seq_b, ground_truth = [], [], [], []
    for record in records:
        row_x, row_a, row_b, row_gt = _abxi_domain_sequences(
            record.sequence, record.domains
        )
        seq_x.append(_left_pad(row_x, length))
        seq_a.append(_left_pad(row_a, length))
        seq_b.append(_left_pad(row_b, length))
        ground_truth.append(_left_pad(row_gt, length))
    return {
        "user_id": torch.tensor([record.user_id for record in records], dtype=torch.long),
        "abxi_sequence": torch.tensor(seq_x, dtype=torch.long),
        "abxi_sequence_a": torch.tensor(seq_a, dtype=torch.long),
        "abxi_sequence_b": torch.tensor(seq_b, dtype=torch.long),
        "abxi_ground_truth": torch.tensor(ground_truth, dtype=torch.long),
        "abxi_full_seen": torch.tensor(
            [_left_pad(record.full_seen_items, seen_length) for record in records],
            dtype=torch.long,
        ),
    }


@dataclass(frozen=True)
class AMIDTrainingRecord:
    user_id: int
    target_item_id: int
    target_domain_id: int
    domain_sequence_0: List[int]
    domain_sequence_1: List[int]
    user_group_id: int


class AMIDTrainingDataset(Dataset[AMIDTrainingRecord]):
    """One row per user and prediction domain, matching official AMID CSV rows."""

    def __init__(
        self,
        corpus: CanonicalCorpus,
        sequence_length: int,
        *,
        included_user_ids: set[int] | None = None,
    ):
        if sequence_length < 1:
            raise ValueError("AMID sequence_length must be positive")
        self.records: List[AMIDTrainingRecord] = []
        pad_item_id = corpus.num_items + 1
        group_ids = {"overlap": 0, "source_only": 1, "target_only": 2, "non_overlap": 3}
        for user_id, events in sorted(corpus.train_sequences.items()):
            if included_user_ids is not None and user_id not in included_user_ids:
                continue
            items_by_domain = {
                domain: [event.item_id for event in events if event.domain_id == domain]
                for domain in range(corpus.num_domains)
            }
            for target_domain in range(corpus.num_domains):
                target_items = items_by_domain[target_domain]
                if not target_items:
                    continue
                sequences = {domain: list(items) for domain, items in items_by_domain.items()}
                target_item = sequences[target_domain].pop()
                self.records.append(
                    AMIDTrainingRecord(
                        user_id=user_id,
                        target_item_id=target_item,
                        target_domain_id=target_domain,
                        domain_sequence_0=_left_pad(
                            sequences[0], sequence_length, pad_item_id
                        ),
                        domain_sequence_1=_left_pad(
                            sequences[1], sequence_length, pad_item_id
                        ),
                        user_group_id=group_ids[corpus.user_groups[user_id]],
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> AMIDTrainingRecord:
        return self.records[index]


def collate_amid_training(records: Sequence[AMIDTrainingRecord]) -> Dict[str, torch.Tensor]:
    if not records:
        raise ValueError("Cannot collate an empty AMID batch")
    return {
        "user_id": torch.tensor([record.user_id for record in records], dtype=torch.long),
        "target_item_id": torch.tensor(
            [record.target_item_id for record in records], dtype=torch.long
        ),
        "target_domain_id": torch.tensor(
            [record.target_domain_id for record in records], dtype=torch.long
        ),
        "domain_item_seq_0": torch.tensor(
            [record.domain_sequence_0 for record in records], dtype=torch.long
        ),
        "domain_item_seq_1": torch.tensor(
            [record.domain_sequence_1 for record in records], dtype=torch.long
        ),
        "user_group_id": torch.tensor(
            [record.user_group_id for record in records], dtype=torch.long
        ),
    }


@dataclass(frozen=True)
class C2DSRTrainingRecord:
    """The 18 tensors emitted by the official C2DSR training loader."""

    user_id: int
    values: tuple[List[int], ...]


def _c2dsr_record(
    user_id: int,
    sequence: Sequence[int],
    *,
    source_item_num: int,
    target_item_num: int,
    max_length: int,
    rng: random.Random,
) -> C2DSRTrainingRecord | None:
    """Canonical-ID equivalent of upstream ``DataLoader.preprocess``.

    C2DSR uses zero-based contiguous item IDs and its final item ID as padding.
    Canonical IDs are already contiguous and domain ordered, so subtracting one
    gives the exact upstream representation.
    """

    # Upstream expects at most maxlen inputs plus one next-item target. Keeping
    # the most recent window is the framework's explicit truncation rule.
    original = [item - 1 for item in sequence[-(max_length + 1) :]]
    if len(original) < 2:
        return None
    item_padding = source_item_num + target_item_num
    source_padding = source_item_num
    target_padding = target_item_num
    ground = list(original[1:])
    share_x_ground = [w if w < source_item_num else source_padding for w in ground]
    share_x_ground_mask = [int(w < source_item_num) for w in ground]
    share_y_ground = [w - source_item_num if w >= source_item_num else target_padding for w in ground]
    share_y_ground_mask = [int(w >= source_item_num) for w in ground]

    seq = list(original[:-1])
    position = list(range(1, len(seq) + 1))
    ground_mask = [1] * len(seq)
    x_seq: List[int] = []
    y_seq: List[int] = []
    x_position: List[int] = []
    y_position: List[int] = []
    corrupt_x: List[int] = []
    corrupt_y: List[int] = []
    x_count = y_count = 1
    for item in seq:
        if item < source_item_num:
            corrupt_x.append(item)
            corrupt_y.append(rng.randint(0, source_item_num - 1))
            x_seq.append(item)
            y_seq.append(item_padding)
            x_position.append(x_count)
            y_position.append(0)
            x_count += 1
        else:
            corrupt_x.append(rng.randint(source_item_num, item_padding - 1))
            corrupt_y.append(item)
            x_seq.append(item_padding)
            y_seq.append(item)
            x_position.append(0)
            y_position.append(y_count)
            y_count += 1

    x_ground = [source_padding] * len(seq)
    x_ground_mask = [0] * len(seq)
    previous = -1
    for reverse_index in range(1, len(seq) + 1):
        index = -reverse_index
        if x_position[index]:
            if previous == -1:
                previous = x_seq[index]
                if ground[-1] < source_item_num:
                    x_ground[index] = ground[-1]
                    x_ground_mask[index] = 1
                else:
                    x_seq[index] = item_padding
                    x_position[index] = 0
            else:
                x_ground[index] = previous
                x_ground_mask[index] = 1
                previous = x_seq[index]
    if not sum(x_ground_mask):
        return None

    y_ground = [target_padding] * len(seq)
    y_ground_mask = [0] * len(seq)
    previous = -1
    for reverse_index in range(1, len(seq) + 1):
        index = -reverse_index
        if y_position[index]:
            if previous == -1:
                previous = y_seq[index] - source_item_num
                # Preserve the upstream strict comparison exactly.
                if ground[-1] > source_item_num:
                    y_ground[index] = ground[-1] - source_item_num
                    y_ground_mask[index] = 1
                else:
                    y_seq[index] = item_padding
                    y_position[index] = 0
            else:
                y_ground[index] = previous
                y_ground_mask[index] = 1
                previous = y_seq[index] - source_item_num
    if not sum(y_ground_mask):
        return None

    padding = max_length - len(seq)
    pads = {
        "item": [item_padding] * padding,
        "source": [source_padding] * padding,
        "target": [target_padding] * padding,
        "zero": [0] * padding,
    }
    values = (
        pads["item"] + seq,
        pads["item"] + x_seq,
        pads["item"] + y_seq,
        pads["zero"] + position,
        pads["zero"] + x_position,
        pads["zero"] + y_position,
        pads["item"] + ground,
        pads["source"] + share_x_ground,
        pads["target"] + share_y_ground,
        pads["source"] + x_ground,
        pads["target"] + y_ground,
        pads["zero"] + ground_mask,
        pads["zero"] + share_x_ground_mask,
        pads["zero"] + share_y_ground_mask,
        pads["zero"] + x_ground_mask,
        pads["zero"] + y_ground_mask,
        pads["item"] + corrupt_x,
        pads["item"] + corrupt_y,
    )
    return C2DSRTrainingRecord(user_id=user_id, values=values)


class C2DSRTrainingDataset(Dataset[C2DSRTrainingRecord]):
    """One upstream-style row per usable canonical training user."""

    def __init__(
        self,
        corpus: CanonicalCorpus,
        sequence_length: int,
        seed: int,
        official_batch_size: int | None = None,
    ):
        if corpus.num_domains != 2:
            raise ValueError("C2DSR requires exactly two domains")
        self.sequence_length = sequence_length
        self.source_item_num = len(corpus.domain_catalogs[0])
        self.target_item_num = len(corpus.domain_catalogs[1])
        rng = random.Random(seed)
        self.records = []
        for user_id, events in sorted(corpus.train_sequences.items()):
            record = _c2dsr_record(
                user_id,
                [event.item_id for event in events],
                source_item_num=self.source_item_num,
                target_item_num=self.target_item_num,
                max_length=sequence_length,
                rng=rng,
            )
            if record is not None:
                self.records.append(record)
        self.num_native_rows = len(self.records)
        # Upstream shuffles once at loader construction, keeps that batch order
        # for every epoch, and repeats leading real rows to complete a remainder.
        if official_batch_size is not None and self.records:
            rng.shuffle(self.records)
            if official_batch_size < len(self.records):
                remainder = len(self.records) % official_batch_size
                if remainder:
                    self.records.extend(self.records[: official_batch_size - remainder])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> C2DSRTrainingRecord:
        return self.records[index]


_C2DSR_FIELDS = (
    "c2_sequence", "c2_sequence_x", "c2_sequence_y", "c2_position",
    "c2_position_x", "c2_position_y", "c2_ground", "c2_share_x_ground",
    "c2_share_y_ground", "c2_x_ground", "c2_y_ground", "c2_ground_mask",
    "c2_share_x_mask", "c2_share_y_mask", "c2_x_mask", "c2_y_mask",
    "c2_corrupt_x", "c2_corrupt_y",
)


def collate_c2dsr_training(records: Sequence[C2DSRTrainingRecord]) -> Dict[str, torch.Tensor]:
    if not records:
        raise ValueError("Cannot collate an empty C2DSR batch")
    batch = {"user_id": torch.tensor([record.user_id for record in records])}
    for index, field in enumerate(_C2DSR_FIELDS):
        batch[field] = torch.tensor([record.values[index] for record in records], dtype=torch.long)
    return batch


@dataclass(frozen=True)
class TriCDRTrainingRecord:
    user_id: int
    mixed_sequence: List[int]
    source_sequence: List[int]
    target_sequence: List[int]
    mixed_indices_for_target: List[int]
    source_indices_for_target: List[int]
    sequence_length: int
    target_catalog: tuple[int, ...]


def _tricdr_alignment(
    mixed: Sequence[int], domains: Sequence[int], source: Sequence[int],
    source_domain_id: int = 0,
) -> tuple[List[int], List[int]]:
    """Exact upstream negative-index alignment used by ``data_partition``."""

    position_mixed = len(mixed) - 1
    position_source = len(source) - 1
    mixed_positions: List[int] = []
    source_positions: List[int] = []
    for domain in reversed(domains):
        if domain == source_domain_id:
            position_source -= 1
        else:
            mixed_positions.append(position_mixed - 1)
            source_positions.append(position_source)
        position_mixed -= 1
    mixed_positions = mixed_positions[:-1]
    source_positions = source_positions[:-1]
    mixed_positions.reverse()
    source_positions.reverse()
    return (
        [position - len(mixed) for position in mixed_positions],
        [position - len(source) for position in source_positions],
    )


class TriCDRTrainingDataset(Dataset[TriCDRTrainingRecord]):
    """Eligible overlap-user pool consumed by the official replacement sampler."""

    def __init__(
        self, corpus: CanonicalCorpus, sequence_length: int, *,
        target_domain_id: int | None = None,
    ):
        if corpus.num_domains != 2:
            raise ValueError("Tri-CDR requires exactly two domains")
        self.target_domain_id = 1 if target_domain_id is None else int(target_domain_id)
        if self.target_domain_id not in (0, 1):
            raise ValueError("Tri-CDR target_domain_id must identify one of two domains")
        self.source_domain_id = 1 - self.target_domain_id
        self.sequence_length = sequence_length
        self.population_count = sum(
            group == "overlap" for group in corpus.user_groups.values()
        )
        target_catalog = tuple(corpus.domain_catalogs[self.target_domain_id])
        self.records: List[TriCDRTrainingRecord] = []
        for user_id, events in sorted(corpus.train_sequences.items()):
            if corpus.user_groups.get(user_id) != "overlap":
                continue
            mixed = [event.item_id for event in events]
            domains = [event.domain_id for event in events]
            source = [
                event.item_id for event in events
                if event.domain_id == self.source_domain_id
            ]
            target = [
                event.item_id for event in events
                if event.domain_id == self.target_domain_id
            ]
            if len(mixed) <= 1 or len(source) <= 1 or len(target) <= 1:
                continue
            mixed_indices, source_indices = _tricdr_alignment(
                mixed, domains, source, self.source_domain_id
            )
            if len(mixed_indices) != len(target) - 1:
                raise ValueError("Tri-CDR target/mixed alignment is inconsistent")
            self.records.append(
                TriCDRTrainingRecord(
                    user_id=user_id,
                    mixed_sequence=mixed,
                    source_sequence=source,
                    target_sequence=target,
                    mixed_indices_for_target=mixed_indices,
                    source_indices_for_target=source_indices,
                    sequence_length=sequence_length,
                    target_catalog=target_catalog,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TriCDRTrainingRecord:
        return self.records[index]


class TriCDRCollator:
    """Stateful equivalent of one official ``WarpSampler`` worker."""

    def __init__(self, seed: int):
        import numpy as np

        self.rng = np.random.RandomState(seed)

    def __call__(self, records: Sequence[TriCDRTrainingRecord]) -> Dict[str, torch.Tensor]:
        if not records:
            raise ValueError("Cannot collate an empty Tri-CDR batch")
        length = records[0].sequence_length
        rows = {key: [] for key in (
            "tricdr_mixed", "tricdr_source", "tricdr_target", "tricdr_positive",
            "tricdr_negative", "tricdr_mixed_indices", "tricdr_source_indices",
        )}
        for record in records:
            mixed = _left_pad(record.mixed_sequence, length)
            source = _left_pad(record.source_sequence, length)
            target_input = record.target_sequence[:-1]
            positives = record.target_sequence[1:]
            seen = set(record.target_sequence)
            negatives = []
            for _ in positives:
                while True:
                    candidate = int(self.rng.choice(record.target_catalog))
                    if candidate not in seen:
                        negatives.append(candidate)
                        break
            mixed_indices = []
            source_indices = []
            for mixed_index, source_index in zip(
                record.mixed_indices_for_target, record.source_indices_for_target
            ):
                mixed_indices.append(0 if mixed_index < -length else mixed_index + length)
                source_indices.append(
                    0
                    if source_index < -length
                    or source_index == -len(record.source_sequence) - 1
                    else source_index + length
                )
            rows["tricdr_mixed"].append(mixed)
            rows["tricdr_source"].append(source)
            rows["tricdr_target"].append(_left_pad(target_input, length))
            rows["tricdr_positive"].append(_left_pad(positives, length))
            rows["tricdr_negative"].append(_left_pad(negatives, length))
            rows["tricdr_mixed_indices"].append(_left_pad(mixed_indices, length))
            rows["tricdr_source_indices"].append(_left_pad(source_indices, length))
        batch = {
            "user_id": torch.tensor([record.user_id for record in records], dtype=torch.long)
        }
        batch.update({key: torch.tensor(value, dtype=torch.long) for key, value in rows.items()})
        return batch


@dataclass(frozen=True)
class SyNCRecTrainingRecord:
    user_id: int
    item_sequence: List[int]
    type_sequence: List[int]
    sequence_length: int
    item_size: int


class SyNCRecTrainingDataset(Dataset[SyNCRecTrainingRecord]):
    """All-user causal rows using the upstream five-token vocabulary offset."""

    def __init__(self, corpus: CanonicalCorpus, sequence_length: int):
        self.records: List[SyNCRecTrainingRecord] = []
        for user_id, events in sorted(corpus.train_sequences.items()):
            if len(events) < 2:
                continue
            self.records.append(
                SyNCRecTrainingRecord(
                    user_id=user_id,
                    item_sequence=[event.item_id + 4 for event in events],
                    type_sequence=[event.domain_id + 5 for event in events],
                    sequence_length=sequence_length,
                    item_size=corpus.num_items + 5,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SyNCRecTrainingRecord:
        return self.records[index]


class SyNCRecCollator:
    """Official CausalDataset negative sampling over the shifted item vocabulary."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def __call__(self, records: Sequence[SyNCRecTrainingRecord]) -> Dict[str, torch.Tensor]:
        if not records:
            raise ValueError("Cannot collate an empty SyNCRec batch")
        rows = {key: [] for key in (
            "syncrec_item_input", "syncrec_item_positive", "syncrec_item_negative",
            "syncrec_type_input", "syncrec_type_positive",
        )}
        for record in records:
            item_input = record.item_sequence[:-1]
            item_positive = record.item_sequence[1:]
            type_input = record.type_sequence[:-1]
            type_positive = record.type_sequence[1:]
            seen = set(record.item_sequence).union({0, 1, 2, 3, 4})
            negatives = []
            for _ in item_input:
                while True:
                    candidate = self.rng.randint(2, record.item_size - 1)
                    if candidate not in seen:
                        negatives.append(candidate)
                        break
            length = record.sequence_length
            rows["syncrec_item_input"].append(_left_pad(item_input, length))
            rows["syncrec_item_positive"].append(_left_pad(item_positive, length))
            rows["syncrec_item_negative"].append(_left_pad(negatives, length))
            rows["syncrec_type_input"].append(_left_pad(type_input, length))
            rows["syncrec_type_positive"].append(_left_pad(type_positive, length))
        batch = {
            "user_id": torch.tensor([record.user_id for record in records], dtype=torch.long)
        }
        batch.update({key: torch.tensor(value, dtype=torch.long) for key, value in rows.items()})
        return batch


@dataclass(frozen=True)
class AutoCDSRTrainingRecord:
    """One upstream MultiSequenceIterable row over the canonical train split."""

    user_id: int
    item_sequence: List[int]
    domain_sequence: List[int]
    sequence_length: int


class AutoCDSRTrainingDataset(Dataset[AutoCDSRTrainingRecord]):
    """All-user sequences with AutoCDSR's reserved sparse-ID prefix."""

    def __init__(self, corpus: CanonicalCorpus, sequence_length: int):
        if corpus.num_domains < 2:
            raise ValueError("AutoCDSR requires at least two domains")
        self.records: List[AutoCDSRTrainingRecord] = []
        for user_id, events in sorted(corpus.train_sequences.items()):
            if not events:
                continue
            self.records.append(
                AutoCDSRTrainingRecord(
                    user_id=user_id,
                    # Upstream reserves 0..99 for padding/mask/IB tokens.
                    item_sequence=[event.item_id + 99 for event in events],
                    domain_sequence=[event.domain_id for event in events],
                    sequence_length=sequence_length,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> AutoCDSRTrainingRecord:
        return self.records[index]


class AutoCDSRCollator:
    """Exact RandomMasking and prefix-truncation semantics from upstream."""

    def __init__(
        self,
        seed: int,
        masking_probability: float = 0.15,
        masking_tolerance: float = 0.05,
        max_trials_for_masking: int = 5,
    ):
        self.generator = torch.Generator().manual_seed(seed)
        self.masking_probability = masking_probability
        self.masking_tolerance = masking_tolerance
        self.max_trials_for_masking = max_trials_for_masking

    @staticmethod
    def _right_pad_prefix(values: Sequence[int], length: int) -> List[int]:
        values = list(values[:length])
        return values + [0] * (length - len(values))

    def __call__(self, records: Sequence[AutoCDSRTrainingRecord]) -> Dict[str, torch.Tensor]:
        if not records:
            raise ValueError("Cannot collate an empty AutoCDSR batch")
        length = records[0].sequence_length
        sequence = torch.tensor(
            [self._right_pad_prefix(record.item_sequence, length) for record in records],
            dtype=torch.long,
        )
        domains = torch.tensor(
            [self._right_pad_prefix(record.domain_sequence, length) for record in records],
            dtype=torch.long,
        )
        content_mask = sequence.ne(0)
        if content_mask.sum(1).eq(0).any():
            raise ValueError("AutoCDSR received an empty sequence")

        proportion = 0.0
        masking_mask = None
        for _ in range(self.max_trials_for_masking + 1):
            masking_mask = (
                torch.rand(sequence.shape, generator=self.generator)
                < self.masking_probability
            ) & content_mask
            proportion = (
                masking_mask.sum(1).float() / content_mask.sum(1).float()
            ).mean().item()
            if (
                self.masking_probability - self.masking_tolerance
                < proportion
                < self.masking_probability + self.masking_tolerance
            ):
                break
        else:
            raise ValueError(
                "Cannot get AutoCDSR's requested masking proportion after "
                f"{self.max_trials_for_masking + 1} trials"
            )
        assert masking_mask is not None
        row_ids = torch.arange(sequence.shape[0])
        forced = torch.multinomial(
            content_mask.float(), num_samples=1, generator=self.generator
        ).squeeze(1)
        masking_mask[row_ids, forced] = True
        labels = sequence[masking_mask].clone()
        label_locations = masking_mask.nonzero()
        masked_sequence = sequence.clone()
        masked_sequence[masking_mask] = 1
        return {
            "user_id": torch.tensor([record.user_id for record in records]),
            "autocdsr_input_ids": masked_sequence,
            "autocdsr_domain_ids": domains,
            "autocdsr_attention_mask": content_mask.long(),
            "autocdsr_labels": labels,
            "autocdsr_label_locations": label_locations,
        }


@dataclass(frozen=True)
class CMGRTrainingRecord:
    user_id: int
    items: List[int]
    domains: List[int]
    sequence_length: int
    source_item_count: int
    target_item_count: int


class CMGRTrainingDataset(Dataset[CMGRTrainingRecord]):
    """All usable canonical users in the upstream seq-to-seq row granularity."""

    def __init__(self, corpus: CanonicalCorpus, sequence_length: int):
        if corpus.num_domains != 2:
            raise ValueError("CMGR requires exactly two domains")
        self.records: List[CMGRTrainingRecord] = []
        source_count = len(corpus.domain_catalogs[0])
        target_count = len(corpus.domain_catalogs[1])
        for user_id, events in sorted(corpus.train_sequences.items()):
            # Upstream requires at least one input/next-item pair. Its original
            # generator accidentally lets length-one rows through, producing
            # empty BCE means (NaN); such rows contain no trainable signal.
            if len(events) < 2:
                continue
            self.records.append(
                CMGRTrainingRecord(
                    user_id=user_id,
                    items=[event.item_id for event in events],
                    domains=[event.domain_id for event in events],
                    sequence_length=sequence_length,
                    source_item_count=source_count,
                    target_item_count=target_count,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CMGRTrainingRecord:
        return self.records[index]


class CMGRCollator:
    """Canonical-ID equivalent of upstream ``CDSRSeq2SeqDataset``."""

    _fields = (
        "cmgr_seq", "cmgr_pos", "cmgr_neg", "cmgr_positions",
        "cmgr_seq_a", "cmgr_pos_a", "cmgr_neg_a", "cmgr_positions_a",
        "cmgr_seq_b", "cmgr_pos_b", "cmgr_neg_b", "cmgr_positions_b",
        "cmgr_target_domain", "cmgr_domain_mask",
    )

    def __init__(self, seed: int):
        import numpy as np

        self.rng = np.random.RandomState(seed)

    def _negative(self, lower: int, upper: int, seen: Sequence[int]) -> int:
        seen_set = set(map(int, seen))
        if upper <= lower or len(seen_set.intersection(range(lower, upper))) >= upper - lower:
            raise ValueError("CMGR negative-sampling range has no unseen item")
        value = int(self.rng.randint(lower, upper))
        while value in seen_set:
            value = int(self.rng.randint(lower, upper))
        return value

    def _truncate(
        self,
        items: Sequence[int],
        domains: Sequence[int],
        length: int,
        item_count_a: int,
        item_count_b: int,
    ) -> tuple[List[int], List[int], List[int], List[int], List[int]]:
        seq = [0] * length
        pos = [0] * length
        neg = [0] * length
        mask = [-1] * length
        if items:
            next_item = int(items[-1])
            index = length - 1
            for reverse_offset, item in enumerate(reversed(items[:-1]), start=2):
                seq[index] = int(item)
                pos[index] = next_item
                sampling_domain = int(domains[-reverse_offset])
                target_domain = int(domains[-reverse_offset + 1])
                if sampling_domain == 0:
                    neg[index] = self._negative(1, item_count_a, items)
                elif sampling_domain == 1:
                    neg[index] = self._negative(
                        item_count_a + 1, item_count_a + item_count_b, items
                    )
                else:
                    raise ValueError("CMGR only supports domain IDs 0 and 1")
                mask[index] = target_domain
                next_item = int(item)
                index -= 1
                if index == -1 or reverse_offset == len(domains):
                    break
        active = sum(value > 0 for value in seq)
        positions = [0] * (length - active) + list(range(1, active + 1))
        return seq, pos, neg, positions, mask

    def _row(self, record: CMGRTrainingRecord):
        n_a, n_b, length = (
            record.source_item_count,
            record.target_item_count,
            record.sequence_length,
        )
        # Canonical IDs already equal the upstream mixed/global IDs. Domain-B
        # local IDs subtract the contiguous source catalog size.
        items = list(record.items)
        domains = list(record.domains)
        mixed = self._truncate(items, domains, length, n_a, n_b)
        window_items = items[-(length + 1) :]
        window_domains = domains[-(length + 1) :]
        local_a = [
            item for item, domain in zip(window_items, window_domains) if domain == 0
        ]
        local_b = [
            item - n_a
            for item, domain in zip(window_items, window_domains)
            if domain == 1
        ]
        view_a = self._truncate(local_a, [0] * len(local_a), length, n_a, n_a)
        view_b = self._truncate(local_b, [0] * len(local_b), length, n_b, n_b)
        domain_mask = mixed[4]
        if window_domains[0] == 0 and 1 in domain_mask:
            domain_mask[domain_mask.index(1)] = -1
        elif window_domains[0] == 1 and 0 in domain_mask:
            domain_mask[domain_mask.index(0)] = -1
        return (
            *mixed[:4],
            *view_a[:4],
            *view_b[:4],
            domains[-1],
            domain_mask,
        )

    def __call__(self, records: Sequence[CMGRTrainingRecord]) -> Dict[str, torch.Tensor]:
        if not records:
            raise ValueError("Cannot collate an empty CMGR batch")
        rows = [self._row(record) for record in records]
        batch = {
            field: torch.tensor([row[index] for row in rows], dtype=torch.long)
            for index, field in enumerate(self._fields)
        }
        batch["user_id"] = torch.tensor([record.user_id for record in records])
        return batch


class HGTLStepDataset(Dataset[int]):
    """Dummy step indices; each collated step draws official replacement batches."""

    def __init__(self, corpus: CanonicalCorpus, batch_size: int):
        self.num_steps = max(1, corpus.num_users // batch_size)

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, index: int) -> int:
        return index


class HGTLStepCollator:
    """Official three-slot ``WarpSampler`` semantics over canonical train data.

    A two-domain corpus retains the historical empty C compatibility slot; a
    three-domain corpus trains all released A/B/C branches without fabricating
    interactions for users missing one or more domains.
    """

    def __init__(self, corpus: CanonicalCorpus, batch_size: int, max_length: int, seed: int):
        import numpy as np

        self.batch_size = batch_size
        self.max_length = max_length
        self.rng = np.random.RandomState(seed)
        if corpus.num_domains not in {2, 3}:
            raise ValueError("HGTL supports two or three canonical domains")
        self.num_domains = corpus.num_domains
        self.item_counts = [
            len(corpus.domain_catalogs[index]) for index in range(self.num_domains)
        ] + [0] * (3 - self.num_domains)
        self.items = {domain: {} for domain in range(self.num_domains)}
        self.times = {domain: {} for domain in range(self.num_domains)}
        for user_id in corpus.user_groups:
            events = corpus.train_sequences.get(user_id, [])
            for domain in range(self.num_domains):
                selected = [event for event in events if event.domain_id == domain]
                self.items[domain][user_id] = [event.item_id for event in selected]
                self.times[domain][user_id] = [int(event.timestamp) for event in selected]
        self.eligible = {
            domain: [
                user_id for user_id, values in self.items[domain].items() if len(values) > 1
            ]
            for domain in range(self.num_domains)
        }
        if any(not users for users in self.eligible.values()):
            raise ValueError("HGTL needs at least one trainable user in each real domain")

    def _negative(self, domain: int, seen: Sequence[int]) -> int:
        lower = sum(self.item_counts[:domain]) + 1
        upper = sum(self.item_counts[: domain + 1]) + 1
        seen_set = set(seen)
        value = int(self.rng.randint(lower, upper))
        while value in seen_set:
            value = int(self.rng.randint(lower, upper))
        return value

    def _aligned_mask(self, main_time, other_time):
        mask = [0] * self.max_length
        other_index = 0
        for index, timestamp in enumerate(main_time):
            while other_index < self.max_length and timestamp >= other_time[other_index]:
                other_index += 1
            mask[index] = other_index
        return mask

    def _sample_one(self, target_domain: int):
        user_id = int(self.rng.choice(self.eligible[target_domain]))
        sequences = [[0] * self.max_length for _ in range(3)]
        timestamps = [[0] * self.max_length for _ in range(3)]
        positives = [0] * self.max_length
        negatives = [0] * self.max_length
        target_items = self.items[target_domain][user_id]
        target_times = self.times[target_domain][user_id]
        next_item = target_items[-1]
        seen = set(target_items)
        index = self.max_length - 1
        for item, timestamp in reversed(list(zip(target_items[:-1], target_times[:-1]))):
            sequences[target_domain][index] = item
            timestamps[target_domain][index] = timestamp
            positives[index] = next_item
            negatives[index] = self._negative(target_domain, seen)
            next_item = item
            index -= 1
            if index == -1:
                break
        other_domains = [domain for domain in range(3) if domain != target_domain]
        for domain in other_domains:
            if domain >= self.num_domains:
                continue
            index = self.max_length - 1
            other_items = self.items[domain][user_id][:-1]
            other_times = self.times[domain][user_id][:-1]
            for item, timestamp in reversed(list(zip(other_items, other_times))):
                sequences[domain][index] = item
                timestamps[domain][index] = timestamp
                index -= 1
                if index == -1:
                    break
        masks = [
            self._aligned_mask(timestamps[target_domain], timestamps[domain])
            for domain in other_domains
        ]
        return user_id, sequences, positives, negatives, masks

    def __call__(self, unused_steps: Sequence[int]) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        for target_domain in range(self.num_domains):
            prefix = f"hgtl_{'abc'[target_domain]}"
            rows = [self._sample_one(target_domain) for _ in range(self.batch_size)]
            result[f"{prefix}_user"] = torch.tensor([row[0] for row in rows])
            for domain, suffix in enumerate(("seq_a", "seq_b", "seq_c")):
                result[f"{prefix}_{suffix}"] = torch.tensor(
                    [row[1][domain] for row in rows], dtype=torch.long
                )
            result[f"{prefix}_pos"] = torch.tensor([row[2] for row in rows])
            result[f"{prefix}_neg"] = torch.tensor([row[3] for row in rows])
            result[f"{prefix}_mask_1"] = torch.tensor([row[4][0] for row in rows])
            result[f"{prefix}_mask_2"] = torch.tensor([row[4][1] for row in rows])
        return result


@dataclass(frozen=True)
class HorizonRecTrainingRecord:
    """One upstream HorizonRec target transition over canonical train events."""

    user_id: int
    target_prefix: List[int]
    source_prefix: List[int]
    mixed_prefix: List[int]
    preference_prefix: List[int]
    target_label: int


class HorizonRecTrainingDataset(Dataset[HorizonRecTrainingRecord]):
    """Canonical equivalent of upstream ``Data_Train.split_onebyone``.

    HorizonRec is a two-domain source-to-target model. Mixed IDs follow its
    released ``toy`` branch: target-local IDs occupy ``[1, T]`` and source
    items occupy ``[T + 1, T + S]``. Each row predicts one target transition;
    source-only users therefore have no native HorizonRec supervision.
    """

    def __init__(
        self, corpus: CanonicalCorpus, sequence_length: int, target_domain_id: int
    ) -> None:
        if corpus.num_domains != 2:
            raise ValueError("HorizonRec requires exactly two domains")
        if sequence_length < 1:
            raise ValueError("HorizonRec sequence_length must be positive")
        self.sequence_length = sequence_length
        self.target_domain_id = int(target_domain_id)
        self.source_domain_id = 1 - self.target_domain_id
        self.target_item_count = len(corpus.domain_catalogs[self.target_domain_id])
        self.source_item_count = len(corpus.domain_catalogs[self.source_domain_id])
        self.records: List[HorizonRecTrainingRecord] = []

        for user_id, events in sorted(corpus.train_sequences.items()):
            target_events = [
                event for event in events if event.domain_id == self.target_domain_id
            ]
            if len(target_events) < 2:
                continue
            mixed_ids = [self._mixed_id(event) for event in events]
            mixed_global_ids = [event.item_id for event in events]
            for index in range(len(target_events) - 1):
                label_event = target_events[index + 1]
                # This intentionally preserves upstream's ``argwhere(...)[-1]``
                # behavior for repeated target items.
                matches = [
                    position
                    for position, item_id in enumerate(mixed_global_ids)
                    if item_id == label_event.item_id
                ]
                if not matches:
                    raise RuntimeError("Target transition is missing from its mixed sequence")
                mixed_index = matches[-1]
                mixed_prefix = mixed_ids[:mixed_index]
                source_prefix = [
                    value - self.target_item_count
                    for value in mixed_prefix
                    if value > self.target_item_count
                ]
                self.records.append(
                    HorizonRecTrainingRecord(
                        user_id=user_id,
                        target_prefix=[
                            event.local_item_id for event in target_events[: index + 1]
                        ],
                        source_prefix=source_prefix,
                        mixed_prefix=mixed_prefix,
                        # Upstream's detached MDR pool includes the label event.
                        preference_prefix=mixed_ids[: mixed_index + 1],
                        target_label=label_event.local_item_id,
                    )
                )
        if not self.records:
            raise ValueError(
                "HorizonRec needs at least one user with two target-domain train events"
            )

    def _mixed_id(self, event) -> int:
        if event.domain_id == self.target_domain_id:
            return int(event.local_item_id)
        return self.target_item_count + int(event.local_item_id)

    @property
    def preference_prefixes(self) -> List[List[int]]:
        return [record.preference_prefix for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> HorizonRecTrainingRecord:
        return self.records[index]


class HorizonRecCollator:
    """Right-pad-to-left layout used by HorizonRec's released TrainDataset."""

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length

    def __call__(
        self, records: Sequence[HorizonRecTrainingRecord]
    ) -> Dict[str, torch.Tensor]:
        if not records:
            raise ValueError("Cannot collate an empty HorizonRec batch")
        return {
            "horizon_target_seq": torch.tensor(
                [_left_pad(record.target_prefix, self.sequence_length) for record in records],
                dtype=torch.long,
            ),
            "horizon_source_seq": torch.tensor(
                [_left_pad(record.source_prefix, self.sequence_length) for record in records],
                dtype=torch.long,
            ),
            "horizon_mixed_seq": torch.tensor(
                [_left_pad(record.mixed_prefix, self.sequence_length) for record in records],
                dtype=torch.long,
            ),
            "horizon_label": torch.tensor(
                [record.target_label for record in records], dtype=torch.long
            ),
            "user_id": torch.tensor([record.user_id for record in records], dtype=torch.long),
        }


class LLM4CDSRTrainingDataset(Dataset[CMGRTrainingRecord]):
    """Official LLM4CDSR row granularity over usable overlapping users.

    The released datasets contain shared users and its per-row item alignment
    loss samples one item from each domain. Canonical non-overlap users remain
    available to the common evaluator, but are not fabricated into paired
    LLM4CDSR training rows.
    """

    def __init__(self, corpus: CanonicalCorpus, sequence_length: int):
        base = CMGRTrainingDataset(corpus, sequence_length)
        self.records = [
            record
            for record in base.records
            if corpus.user_groups.get(record.user_id) == "overlap"
            # The released regularizer removes the final event before sampling
            # one item from each domain, so both domains must occur beforehand.
            and set(record.domains[:-1]) == {0, 1}
        ]
        if not self.records:
            raise ValueError(
                "LLM4CDSR needs at least one overlapping train sequence containing both domains"
            )
        # Upstream passes the zero-based dataset row index to user_emb_llm.
        self.profile_index_by_user = {
            record.user_id: index for index, record in enumerate(self.records)
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CMGRTrainingRecord:
        return self.records[index]


class LLM4CDSRCollator(CMGRCollator):
    """CDSRRegSeq2SeqDatasetUser semantics plus canonical user alignment."""

    def __init__(self, seed: int, profile_index_by_user: Dict[int, int]):
        super().__init__(seed)
        self.profile_index_by_user = dict(profile_index_by_user)

    def __call__(self, records: Sequence[CMGRTrainingRecord]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(records)
        reg_a, reg_b, user_indices = [], [], []
        for record in records:
            # Official code excludes the final event before drawing one item
            # per domain with NumPy choice.
            paired = list(zip(record.items[:-1], record.domains[:-1]))
            candidates_a = [item for item, domain in paired if domain == 0]
            candidates_b = [item for item, domain in paired if domain == 1]
            if not candidates_a or not candidates_b:
                raise ValueError("LLM4CDSR paired regularization received a one-domain row")
            reg_a.append(int(self.rng.choice(candidates_a)))
            reg_b.append(int(self.rng.choice(candidates_b)))
            user_indices.append(self.profile_index_by_user[record.user_id])
        batch["llm_reg_a"] = torch.tensor(reg_a, dtype=torch.long)
        batch["llm_reg_b"] = torch.tensor(reg_b, dtype=torch.long)
        batch["llm_user_index"] = torch.tensor(user_indices, dtype=torch.long)
        return batch


class SemaCDRStepDataset(Dataset[int]):
    """Epoch-sized step view used by the released SemaCDR sampler."""

    def __init__(self, corpus: CanonicalCorpus, target_domain_id: int, batch_size: int):
        target_lengths = [
            sum(event.domain_id == target_domain_id for event in events)
            for events in corpus.train_sequences.values()
        ]
        if not any(length >= 2 for length in target_lengths):
            raise ValueError("SemaCDR needs a user with two target-domain train events")
        self.raw_steps = sum(target_lengths) // batch_size
        self.steps = max(1, self.raw_steps)

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int) -> int:
        return index


class SemaCDRStepCollator:
    """Fixed replacement batch with the upstream target/source truncation rules."""

    def __init__(self, corpus: CanonicalCorpus, target_domain_id: int, batch_size: int,
                 max_length: int, seed: int):
        if corpus.num_domains != 2:
            raise ValueError("SemaCDR requires exactly two domains")
        self.target_domain_id = target_domain_id
        self.source_domain_id = 1 - target_domain_id
        self.target_count = len(corpus.domain_catalogs[target_domain_id])
        self.batch_size = batch_size
        self.max_length = max_length
        self.rng = random.Random(seed)
        self.target_items: Dict[int, List[int]] = {}
        self.target_times: Dict[int, List[float]] = {}
        self.source_items: Dict[int, List[int]] = {}
        self.source_times: Dict[int, List[float]] = {}
        for user_id, events in corpus.train_sequences.items():
            targets = [event for event in events if event.domain_id == target_domain_id]
            sources = [event for event in events if event.domain_id == self.source_domain_id]
            if targets:
                self.target_items[user_id] = [event.local_item_id for event in targets]
                self.target_times[user_id] = [event.timestamp for event in targets]
            if sources:
                self.source_items[user_id] = [
                    self.target_count + event.local_item_id for event in sources
                ]
                self.source_times[user_id] = [event.timestamp for event in sources]
        self.eligible = sorted(
            user_id for user_id, values in self.target_items.items() if len(values) >= 2
        )

    @staticmethod
    def build_seq_mix(target: Sequence[int], source: Sequence[int], max_length: int):
        result = list(target)
        target_length = sum(value != 0 for value in result)
        if target_length >= max_length:
            return result
        source_values = [value for value in source if value != 0]
        take = source_values[-(max_length - target_length):]
        start = max_length - target_length - len(take)
        result[:max_length - target_length] = [0] * (max_length - target_length)
        result[start:max_length - target_length] = take
        return result

    def _negative(self, seen: set[int]) -> int:
        available = self.target_count - len(seen.intersection(range(1, self.target_count + 1)))
        if available <= 0:
            raise ValueError("SemaCDR cannot sample an unseen target item")
        value = self.rng.randint(1, self.target_count)
        while value in seen:
            value = self.rng.randint(1, self.target_count)
        return value

    def _sample(self):
        user_id = self.rng.choice(self.eligible)
        items = self.target_items[user_id]
        times = self.target_times[user_id]
        selected = items[-(self.max_length - 1):]
        selected_times = times[-len(selected):]
        input_length = len(selected) - 1
        seq = [0] * self.max_length
        pos = [0] * self.max_length
        neg = [0] * self.max_length
        target_times = [0.0] * self.max_length
        start = self.max_length - input_length
        seq[start:] = selected[:-1]
        pos[start:] = selected[1:]
        target_times[start:start + input_length] = selected_times[1:1 + input_length]
        seen = set(items)
        for index in range(start, self.max_length):
            if pos[index]:
                neg[index] = self._negative(seen)
        source = _left_pad(self.source_items.get(user_id, []), self.max_length)
        source_times = _left_pad(self.source_times.get(user_id, []), self.max_length, 0)
        mask, source_index = [0] * self.max_length, 0
        for index in range(self.max_length):
            while (source_index < self.max_length and target_times[index] > 0
                   and source_times[source_index] > 0
                   and target_times[index] >= source_times[source_index]):
                source_index += 1
            mask[index] = source_index
        mixed = self.build_seq_mix(seq, source, self.max_length) if user_id in self.source_items else seq
        return user_id, seq, pos, neg, source, mask, mixed

    def __call__(self, unused_steps: Sequence[int]) -> Dict[str, torch.Tensor]:
        rows = [self._sample() for _ in range(self.batch_size)]
        names = ("sema_user", "sema_seq", "sema_pos", "sema_neg", "sema_source_seq",
                 "sema_mask", "sema_mix_seq")
        return {
            name: torch.tensor([row[index] for row in rows], dtype=torch.long)
            for index, name in enumerate(names)
        }
