"""Immutable split and candidate protocol shared by every baseline."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from cdsr.data.schema import CanonicalCorpus, SequenceExample


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def example_id(
    example: SequenceExample,
    *,
    raw_user_id: str | None = None,
    raw_target_item_id: str | None = None,
) -> str:
    """Stable example identity, optionally independent of canonical ID shifts.

    Adding an auxiliary source domain changes sorted canonical user/item IDs.
    Raw identities keep target-domain sampling identical between the two-domain
    and multi-domain views of the same Books/Movies example.
    """
    identity = {
        "user_id": raw_user_id if raw_user_id is not None else example.user_id,
        "target_item_id": (
            raw_target_item_id
            if raw_target_item_id is not None
            else example.target_item_id
        ),
        "target_timestamp": example.target_timestamp,
        "split": example.split,
    }
    return _digest(identity)[:24]


def corpus_fingerprint(corpus: CanonicalCorpus) -> str:
    value = {
        "user_to_id": corpus.user_to_id,
        "domain_to_id": corpus.domain_to_id,
        "global_item_to_id": corpus.global_item_to_id,
        "domain_catalogs": corpus.domain_catalogs,
        "metadata": corpus.metadata,
        "examples": {
            split: [example.to_dict() for example in examples]
            for split, examples in corpus.examples.items()
        },
    }
    return _digest(value)


@dataclass(frozen=True)
class ProtocolConfig:
    candidate_mode: str = "sampled"
    num_negatives: int = 999
    filter_history: bool = True
    seed: int = 2026
    topk: tuple[int, ...] = (5, 10, 20)
    target_domain_ids: tuple[int, ...] | None = None
    require_exact_num_negatives: bool = True

    def __post_init__(self) -> None:
        if self.candidate_mode not in {"sampled", "full"}:
            raise ValueError("candidate_mode must be sampled or full")
        if self.num_negatives < 1:
            raise ValueError("num_negatives must be positive")
        if not self.topk or min(self.topk) < 1:
            raise ValueError("topk values must be positive")


@dataclass
class ProtocolRow:
    example_id: str
    split: str
    user_id: int
    target_item_id: int
    target_domain_id: int
    target_timestamp: float
    user_group: str
    candidate_ids: List[int]
    target_index: int
    candidate_fingerprint: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ProtocolBundle:
    path: Path
    manifest: Dict[str, object]
    rows: Dict[str, List[ProtocolRow]]

    @property
    def protocol_id(self) -> str:
        return str(self.manifest["protocol_id"])


def _make_candidates(
    example: SequenceExample,
    catalog: Sequence[int],
    config: ProtocolConfig,
    row_id: str,
    raw_item_by_id: Mapping[int, str],
) -> List[int]:
    # Candidates belong to the target domain, so only target-domain history is
    # relevant.  Using the truncated mixed history would make adding CD events
    # change which previously watched Movies are filtered out.
    history = set(example.target_item_seq) if config.filter_history else set()
    # Canonical global IDs move when an auxiliary domain is inserted before the
    # target domain.  Sort by raw domain/item identity before sampling so the
    # frozen Books -> Movies candidates remain identical in the three-domain
    # Books + CD -> Movies view.
    forbidden = history | {example.target_item_id}
    eligible_count = len(catalog) - sum(item in forbidden for item in catalog)
    seed = config.seed + int(row_id[:16], 16)
    rng = random.Random(seed)
    if config.candidate_mode == "sampled":
        if config.require_exact_num_negatives and eligible_count < config.num_negatives:
            raise ValueError(
                f"Example {row_id} has only {eligible_count} eligible negatives; "
                f"protocol requires exactly {config.num_negatives}"
            )
        sample_size = min(config.num_negatives, eligible_count)
        if sample_size * 4 >= eligible_count:
            eligible = [item for item in catalog if item not in forbidden]
            negatives = rng.sample(eligible, sample_size)
        else:
            # Uniform rejection sampling is equivalent to sampling the eligible
            # catalog without replacement, while avoiding an O(catalog_size)
            # allocation for every evaluation row.
            selected: set[int] = set()
            negatives = []
            while len(negatives) < sample_size:
                item = catalog[rng.randrange(len(catalog))]
                if item in forbidden or item in selected:
                    continue
                selected.add(item)
                negatives.append(item)
    else:
        negatives = [item for item in catalog if item not in forbidden]
    candidates = [example.target_item_id, *negatives]
    # Randomize target position so text/generative adapters cannot exploit a
    # fixed positive-at-index-zero convention.
    rng.shuffle(candidates)
    return candidates


def build_protocol(
    corpus: CanonicalCorpus,
    output_dir: str | Path,
    config: ProtocolConfig | None = None,
) -> ProtocolBundle:
    protocol_config = config or ProtocolConfig()
    destination = Path(output_dir)
    fingerprint = corpus_fingerprint(corpus)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        existing = load_protocol(destination)
        same_corpus = existing.manifest.get("corpus_fingerprint") == fingerprint
        same_config = _canonical_json(existing.manifest.get("config")) == _canonical_json(
            asdict(protocol_config)
        )
        if not same_corpus or not same_config:
            raise ValueError(
                f"Refusing to overwrite immutable protocol at {destination}; "
                "use a new protocol_dir for different data or candidate settings"
            )
        validate_protocol_against_corpus(existing, corpus)
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, List[ProtocolRow]] = {"valid": [], "test": []}
    seen_ids: set[str] = set()
    raw_user_by_id = {value: key for key, value in corpus.user_to_id.items()}
    raw_item_by_id = {value: key for key, value in corpus.global_item_to_id.items()}
    stable_catalogs = {
        domain_id: sorted(catalog, key=raw_item_by_id.__getitem__)
        for domain_id, catalog in corpus.domain_catalogs.items()
    }
    for split in rows:
        for example in corpus.examples[split]:
            if (
                protocol_config.target_domain_ids is not None
                and example.target_domain_id not in protocol_config.target_domain_ids
            ):
                continue
            row_id = example_id(
                example,
                raw_user_id=raw_user_by_id[example.user_id],
                raw_target_item_id=raw_item_by_id[example.target_item_id],
            )
            if row_id in seen_ids:
                raise ValueError(f"Duplicate protocol example identity: {row_id}")
            seen_ids.add(row_id)
            candidates = _make_candidates(
                example,
                stable_catalogs[example.target_domain_id],
                protocol_config,
                row_id,
                raw_item_by_id,
            )
            row = ProtocolRow(
                example_id=row_id,
                split=split,
                user_id=example.user_id,
                target_item_id=example.target_item_id,
                target_domain_id=example.target_domain_id,
                target_timestamp=example.target_timestamp,
                user_group=example.user_group,
                candidate_ids=candidates,
                target_index=candidates.index(example.target_item_id),
                candidate_fingerprint=_digest(candidates),
            )
            rows[split].append(row)

    manifest_without_id: Dict[str, object] = {
        "version": 1,
        "corpus_fingerprint": fingerprint,
        "config": asdict(protocol_config),
        "split_counts": {split: len(values) for split, values in rows.items()},
        "candidate_files": {split: f"candidates.{split}.jsonl" for split in rows},
    }
    manifest = dict(manifest_without_id)
    manifest["protocol_id"] = _digest(
        {
            **manifest_without_id,
            "rows": {split: [row.to_dict() for row in values] for split, values in rows.items()},
        }
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for split, values in rows.items():
        with (destination / f"candidates.{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    return ProtocolBundle(destination, manifest, rows)


def load_protocol(path: str | Path) -> ProtocolBundle:
    source = Path(path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    rows: Dict[str, List[ProtocolRow]] = {}
    for split, filename in manifest["candidate_files"].items():
        rows[split] = []
        with (source / filename).open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows[split].append(ProtocolRow(**json.loads(line)))
    rebuilt = _digest(
        {
            **{key: value for key, value in manifest.items() if key != "protocol_id"},
            "rows": {split: [row.to_dict() for row in values] for split, values in rows.items()},
        }
    )
    if rebuilt != manifest["protocol_id"]:
        raise ValueError("Protocol fingerprint mismatch; candidate files or manifest were modified")
    return ProtocolBundle(source, manifest, rows)


def validate_protocol_against_corpus(bundle: ProtocolBundle, corpus: CanonicalCorpus) -> None:
    if bundle.manifest["corpus_fingerprint"] != corpus_fingerprint(corpus):
        raise ValueError("Protocol was built from a different canonical corpus")
    raw_user_by_id = {value: key for key, value in corpus.user_to_id.items()}
    raw_item_by_id = {value: key for key, value in corpus.global_item_to_id.items()}
    examples = {
        example_id(
            example,
            raw_user_id=raw_user_by_id[example.user_id],
            raw_target_item_id=raw_item_by_id[example.target_item_id],
        ): example
        for split in ("valid", "test")
        for example in corpus.examples[split]
    }
    for split, rows in bundle.rows.items():
        for row in rows:
            example = examples.get(row.example_id)
            if example is None or example.split != split:
                raise ValueError(f"Unknown {split} example {row.example_id}")
            if row.candidate_ids.count(row.target_item_id) != 1:
                raise ValueError(f"Target must occur exactly once for {row.example_id}")
            if row.user_group != example.user_group:
                raise ValueError(f"User group mismatch for {row.example_id}")
            catalog = set(corpus.domain_catalogs[row.target_domain_id])
            if not set(row.candidate_ids).issubset(catalog):
                raise ValueError(f"Cross-domain candidate detected for {row.example_id}")
            if bool(bundle.manifest["config"]["filter_history"]):
                forbidden = set(example.target_item_seq) - {row.target_item_id}
                if forbidden.intersection(row.candidate_ids):
                    raise ValueError(f"History item leaked into candidates for {row.example_id}")
