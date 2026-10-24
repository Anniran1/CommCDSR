"""RecBole-style layered YAML configuration for unified CDSR experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import yaml

from cdsr.data import RawInteraction, build_canonical_corpus, read_domain_interactions, save_corpus
from cdsr.protocol import ProtocolConfig, build_protocol


_VARIABLE = re.compile(r"\$\{([a-zA-Z0-9_.-]+)\}")


def _deep_merge(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Top-level YAML value must be a mapping: {path}")
    return value


def _lookup(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Unknown configuration variable {dotted!r}")
        value = value[part]
    return value


def _resolve_variables(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_variables(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_variables(item, context) for item in value]
    if not isinstance(value, str):
        return value
    matches = list(_VARIABLE.finditer(value))
    if not matches:
        return value
    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        return copy.deepcopy(_lookup(context, matches[0].group(1)))
    output = value
    for match in matches:
        output = output.replace(match.group(0), str(_lookup(context, match.group(1))))
    return output


def _redact_secrets(value: Any) -> Any:
    """Return a serializable config copy with credentials removed."""

    if isinstance(value, Mapping):
        return {
            key: ("" if str(key).lower().endswith("api_key")
                  or str(key).lower() in {"access_token", "secret_key"}
                  else _redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ExperimentConfig:
    values: Dict[str, Any]
    sources: Dict[str, str]
    config_id: str

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return _lookup(self.values, dotted)
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.values)


def load_experiment_config(
    *,
    overall: str | Path,
    dataset: str | Path,
    model: str | Path,
    project_root: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    paths = {
        "overall": Path(overall).resolve(),
        "dataset": Path(dataset).resolve(),
        "model": Path(model).resolve(),
    }
    overall_values = _read_yaml(paths["overall"])
    dataset_values = _read_yaml(paths["dataset"])
    model_values = _read_yaml(paths["model"])

    forbidden_model_sections = {"data", "protocol", "evaluation"}.intersection(model_values)
    if forbidden_model_sections:
        raise ValueError(
            "Model YAML cannot override fairness-critical sections: "
            f"{sorted(forbidden_model_sections)}"
        )
    merged: Dict[str, Any] = {}
    _deep_merge(merged, overall_values)
    _deep_merge(merged, dataset_values)
    _deep_merge(merged, model_values)
    if overrides:
        _deep_merge(merged, overrides)

    root = Path(project_root).resolve() if project_root else paths["overall"].parents[1]
    merged.setdefault("paths", {})
    merged["paths"]["project_root"] = str(root)
    # Resolve repeatedly because dataset paths may reference another resolved key.
    for _ in range(3):
        merged = _resolve_variables(merged, merged)
    validate_experiment_config(merged)
    normalized = json.dumps(merged, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return ExperimentConfig(
        values=merged,
        sources={key: str(value) for key, value in paths.items()},
        config_id=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    for section in ("experiment", "paths", "data", "protocol", "evaluation", "training", "model"):
        if section not in config:
            raise ValueError(f"Missing configuration section {section!r}")
    data = config["data"]
    if not data.get("source_domains"):
        raise ValueError("At least one source domain is required")
    if not data.get("target_domain"):
        raise ValueError("One target domain is required")
    names = [entry["name"] for entry in data["source_domains"]] + [data["target_domain"]["name"]]
    if len(names) != len(set(names)):
        raise ValueError("Domain names must be unique")
    if data.get("user_scope", "overlap") not in {"overlap", "all"}:
        raise ValueError("data.user_scope must be overlap or all")
    if config["protocol"]["candidate_mode"] not in {"sampled", "full"}:
        raise ValueError("Unsupported protocol.candidate_mode")
    if config["protocol"].get("target_domain_only") is not True:
        raise ValueError("Unified comparison requires protocol.target_domain_only: true")
    if config["protocol"].get("shared_candidate_file") is not True:
        raise ValueError("Unified comparison requires protocol.shared_candidate_file: true")
    if config["evaluation"].get("metric_implementation") != "cdsr":
        raise ValueError("All protocol experiments must use the cdsr metric implementation")
    minimum_sources = int(config.get("resources", {}).get("minimum_source_domains", 1))
    actual_sources = len(data["source_domains"])
    if actual_sources < minimum_sources:
        raise ValueError(
            f"Model {config['model']['name']} requires at least {minimum_sources} source domains, "
            f"but dataset {config.get('dataset', {}).get('name', '<unnamed>')} provides "
            f"{actual_sources}"
        )


def _read_configured_domain(entry: Mapping[str, Any], row_offset: int) -> list[RawInteraction]:
    columns = entry.get("columns", {})
    filters = entry.get("filters", {})
    return read_domain_interactions(
        entry["interaction_file"],
        domain=entry["name"],
        user_col=columns.get("user", "user_id"),
        item_col=columns.get("item", "item_id"),
        timestamp_col=columns.get("timestamp", "timestamp"),
        rating_col=columns.get("rating"),
        min_rating=filters.get("min_rating"),
        delimiter=entry.get("delimiter", "\t"),
        row_offset=row_offset,
    )


def load_configured_interactions(config: ExperimentConfig) -> list[RawInteraction]:
    data = config["data"]
    source_entries = list(data["source_domains"])
    target_entry = data["target_domain"]
    interactions: list[RawInteraction] = []
    row_offset = 0
    users_by_domain: Dict[str, set[str]] = {}
    for entry in [*source_entries, target_entry]:
        rows = _read_configured_domain(entry, row_offset)
        row_offset += len(rows) + 10
        interactions.extend(rows)
        users_by_domain[entry["name"]] = {row.user for row in rows}
    if data.get("user_scope", "overlap") == "overlap":
        permitted_users = set.intersection(*users_by_domain.values())
        interactions = [row for row in interactions if row.user in permitted_users]
    if not interactions:
        raise ValueError("No interactions remain after configured filtering")
    return interactions


def prepare_experiment_data(config: ExperimentConfig) -> Dict[str, Any]:
    interactions = load_configured_interactions(config)
    data = config["data"]
    corpus = build_canonical_corpus(
        interactions,
        max_sequence_length=int(data["max_sequence_length"]),
        min_history=int(data["min_history"]),
        min_domain_interactions=int(data["min_domain_interactions"]),
        split_strategy=str(data["split_strategy"]),
        target_domain=str(data["target_domain"]["name"]),
        split_anchor_domains=data.get("split_anchor_domains"),
    )
    canonical_dir = Path(config["paths"]["canonical_dir"])
    protocol_dir = Path(config["paths"]["protocol_dir"])
    save_corpus(corpus, canonical_dir)
    target_name = data["target_domain"]["name"]
    target_id = corpus.domain_to_id[target_name]
    protocol_values = config["protocol"]
    bundle = build_protocol(
        corpus,
        protocol_dir,
        ProtocolConfig(
            candidate_mode=protocol_values["candidate_mode"],
            num_negatives=int(protocol_values["num_negatives"]),
            filter_history=bool(protocol_values["filter_history"]),
            seed=int(protocol_values["seed"]),
            topk=tuple(map(int, config["evaluation"]["topk"])),
            target_domain_ids=(target_id,),
            require_exact_num_negatives=bool(
                protocol_values.get("require_exact_num_negatives", True)
            ),
        ),
    )
    resolved_config = _redact_secrets(config.to_dict())
    resolved_config["config_id"] = config.config_id
    experiment_dir = Path(config["paths"]["output_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return {
        "config_id": config.config_id,
        "canonical_dir": str(canonical_dir),
        "protocol_dir": str(protocol_dir),
        "protocol_id": bundle.protocol_id,
        "resolved_config": str(experiment_dir / "resolved_config.yaml"),
        "users": corpus.num_users,
        "items": corpus.num_items,
        "domains": corpus.num_domains,
        "domain_to_id": corpus.domain_to_id,
        "user_group_counts": corpus.metadata.get("user_group_counts", {}),
        "examples": {split: len(values) for split, values in corpus.examples.items()},
        "evaluation_examples": {split: len(values) for split, values in bundle.rows.items()},
        "candidate_sizes": {
            split: {
                "min": min((len(row.candidate_ids) for row in rows), default=0),
                "max": max((len(row.candidate_ids) for row in rows), default=0),
            }
            for split, rows in bundle.rows.items()
        },
    }
