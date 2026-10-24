"""Canonical, framework-owned data structures for CDSR.

IDs use these conventions throughout the framework:

* ``0`` is reserved for item/user padding.
* domain IDs are zero based.
* item IDs are globally unique across domains.
* local item IDs are one based inside a domain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass(frozen=True)
class RawInteraction:
    user: str
    item: str
    domain: str
    timestamp: float
    row_number: int = 0


@dataclass(frozen=True)
class CanonicalInteraction:
    user_id: int
    item_id: int
    local_item_id: int
    domain_id: int
    timestamp: float
    row_number: int = 0


@dataclass
class SequenceExample:
    user_id: int
    mixed_item_seq: List[int]
    mixed_local_item_seq: List[int]
    mixed_domain_seq: List[int]
    mixed_timestamp_seq: List[float]
    source_item_seq: List[int]
    target_item_seq: List[int]
    target_item_id: int
    target_local_item_id: int
    target_domain_id: int
    target_timestamp: float
    split: str
    user_group: str = "overlap"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SequenceExample":
        return cls(**dict(value))


@dataclass
class CanonicalCorpus:
    examples: Dict[str, List[SequenceExample]]
    user_to_id: Dict[str, int]
    domain_to_id: Dict[str, int]
    global_item_to_id: Dict[str, int]
    local_item_to_id: Dict[str, Dict[str, int]]
    domain_catalogs: Dict[int, List[int]]
    train_sequences: Dict[int, List[CanonicalInteraction]] = field(default_factory=dict)
    user_groups: Dict[int, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_users(self) -> int:
        return len(self.user_to_id)

    @property
    def num_items(self) -> int:
        return len(self.global_item_to_id)

    @property
    def num_domains(self) -> int:
        return len(self.domain_to_id)
