"""Intrinsic and recommendation-level validation for the three CommCDSR modules."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from cdsr.data.schema import CanonicalCorpus
from cdsr.features.comm_cdsr_best import (
    CommCDSRFeatures,
    _item_metadata,
    build_comm_cdsr_hierarchy,
    build_gnn_modularity_hierarchy,
    training_prefix_corpus,
)
from cdsr.models.comm_cdsr_best import CommCDSRModel


def _ari(left: torch.Tensor, right: torch.Tensor) -> float:
    pairs = Counter(zip(left.tolist(), right.tolist()))
    rows, cols = Counter(left.tolist()), Counter(right.tolist())
    choose2 = lambda value: value * (value - 1) / 2
    total = choose2(len(left))
    if total == 0:
        return 1.0
    index = sum(choose2(value) for value in pairs.values())
    row_sum = sum(choose2(value) for value in rows.values())
    col_sum = sum(choose2(value) for value in cols.values())
    expected = row_sum * col_sum / total
    maximum = 0.5 * (row_sum + col_sum)
    return float((index - expected) / max(maximum - expected, 1e-12))


def _nmi(left: torch.Tensor, right: torch.Tensor) -> float:
    pairs = Counter(zip(left.tolist(), right.tolist()))
    rows, cols, count = Counter(left.tolist()), Counter(right.tolist()), len(left)
    mutual = sum(
        value / count * math.log((value * count) / (rows[a] * cols[b]))
        for (a, b), value in pairs.items() if value
    )
    entropy_left = -sum(value / count * math.log(value / count) for value in rows.values())
    entropy_right = -sum(value / count * math.log(value / count) for value in cols.values())
    return float(2 * mutual / max(entropy_left + entropy_right, 1e-12))


def _js(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().add(1e-12); left /= left.sum()
    right = right.float().add(1e-12); right /= right.sum()
    middle = 0.5 * (left + right)
    return float(0.5 * ((left * (left / middle).log()).sum()
                        + (right * (right / middle).log()).sum()))


def _pairwise(values: Sequence[torch.Tensor]) -> list[float]:
    return [_js(values[i], values[j]) for i in range(len(values))
            for j in range(i + 1, len(values))]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _tokens(value: Any) -> set[str]:
    text = " ".join(map(str, value)) if isinstance(value, list) else str(value)
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {
        "the", "and", "for", "with", "from", "into", "that", "this", "of", "to", "a",
    }


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result = []
        for key, item in value.items():
            if key not in {"community_id", "domain", "schema_version", "parent_community_id"}:
                result.extend(_flatten_strings(item))
        return result
    if isinstance(value, list):
        return [text for item in value for text in _flatten_strings(item)]
    return []


def _community_intrinsics(
    corpus: CanonicalCorpus, features: CommCDSRFeatures,
    domain_entries: Sequence[Mapping[str, Any]], settings: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    communities = features.memberships.shape[-1]
    coarse = features.fine_to_coarse.shape[-1]
    if str(settings.get("community_assignment")) == "gnn_modularity":
        builds = [build_gnn_modularity_hierarchy(
            corpus, seed=seed,
            feature_dim=int(settings.get("discovery_feature_dim", 128)),
            gnn_epochs=int(settings.get("gnn_epochs", 12)),
            gnn_layers=int(settings.get("gnn_layers", 2)),
            modularity_resolution=float(settings.get("modularity_resolution", 1.0)),
            minimum_community_size=(
                int(settings["minimum_community_size"])
                if settings.get("minimum_community_size") is not None else None
            ),
        ) for seed in seeds]
    else:
        builds = [build_comm_cdsr_hierarchy(
            corpus, communities, coarse, seed=seed,
            feature_dim=int(settings.get("discovery_feature_dim", 256)),
            temperature=float(settings.get("discovery_temperature", 0.2)),
            assignment="hard_spherical_kmeans",
            clustering_iterations=int(settings.get("clustering_iterations", 50)),
            min_cluster_fraction=float(settings.get("min_cluster_fraction", 0.01)),
        ) for seed in seeds]
    metadata = _item_metadata(corpus, domain_entries)
    domains = []
    for domain in range(corpus.num_domains):
        active = features.memberships[domain].sum(dim=1).gt(0)
        labels = features.memberships[domain, active].argmax(dim=1)
        present = features.memberships[domain].sum(dim=0).gt(0)
        sizes = labels.bincount(minlength=communities)[present]
        stability = []
        for build in builds:
            other_active = build["memberships"][domain].sum(dim=1).gt(0)
            common = active & other_active
            alternate = build["memberships"][domain, common].argmax(dim=1)
            reference = features.memberships[domain, common].argmax(dim=1)
            stability.append({"seed": seeds[len(stability)], "ari": _ari(reference, alternate),
                              "nmi": _nmi(reference, alternate)})
        fused = builds[0].get("fused_user_features")
        if fused is None:
            fused = F.normalize(
                0.5 * builds[0]["collaborative_user_features"]
                + 0.5 * builds[0]["transition_user_features"], dim=2,
            )
        fused = fused[domain]
        # Compare states and centroids in the same fitted coordinate system.
        centroids = F.normalize(torch.einsum(
            "uk,uh->kh", features.memberships[domain], fused
        ), dim=1)
        intra = 1.0 - (fused[active] * centroids[labels]).sum(dim=1)
        inter_matrix = 1.0 - centroids @ centroids.T
        off_diagonal = ~torch.eye(communities, dtype=torch.bool) & present[:, None] & present[None, :]

        category_names = sorted({metadata[event.item_id]["category"]
                                 for events in corpus.train_sequences.values()
                                 for event in events if event.domain_id == domain})
        category_index = {name: index for index, name in enumerate(category_names)}
        category_dist = [torch.zeros(len(category_names)) for _ in range(communities)]
        transition_dist = [Counter() for _ in range(communities)]
        for user_id, events in corpus.train_sequences.items():
            local = [event for event in events if event.domain_id == domain]
            if not local or not active[user_id]:
                continue
            community = int(features.memberships[domain, user_id].argmax())
            for event in local:
                category_dist[community][category_index[metadata[event.item_id]["category"]]] += 1
            for left, right in zip(local, local[1:]):
                transition_dist[community][(
                    metadata[left.item_id]["category"], metadata[right.item_id]["category"]
                )] += 1
        transition_keys = sorted({key for counter in transition_dist for key in counter})
        transition_vectors = [torch.tensor([counter[key] for key in transition_keys], dtype=torch.float32)
                              for counter in transition_dist]
        domains.append({
            "domain": domain_entries[domain]["name"],
            "community_sizes": sizes.tolist(),
            "size_summary": {"min": int(sizes.min()), "max": int(sizes.max()),
                             "mean": float(sizes.float().mean()),
                             "std": float(sizes.float().std(unbiased=False))},
            "seed_stability": stability,
            "mean_intra_cosine_distance": float(intra.mean()),
            "mean_inter_centroid_cosine_distance": float(inter_matrix[off_diagonal].mean()) if off_diagonal.any() else 0.0,
            "mean_pairwise_category_js_divergence": _mean(_pairwise([v for i, v in enumerate(category_dist) if present[i]])),
            "mean_pairwise_transition_js_divergence": _mean(_pairwise([v for i, v in enumerate(transition_vectors) if present[i]])),
        })
    return {"seeds": list(seeds), "domains": domains,
            "identity_used_for_formation_or_bridge": False}


def _community_only_next_item(corpus: CanonicalCorpus, features: CommCDSRFeatures,
                              target_domain: int, ks: Sequence[int],
                              settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or {}
    prefix, excluded = training_prefix_corpus(corpus)
    build = build_gnn_modularity_hierarchy(
        prefix, seed=int(settings.get("audit_seed", 2022)),
        feature_dim=int(settings.get("discovery_feature_dim", 128)),
        gnn_epochs=int(settings.get("gnn_epochs", 12)),
        gnn_layers=int(settings.get("gnn_layers", 2)),
        modularity_resolution=float(settings.get("modularity_resolution", 1.0)),
        select_resolution=False,
    )
    memberships = build["memberships"]
    catalog = corpus.domain_catalogs[target_domain]
    catalog_index = {item: index for index, item in enumerate(catalog)}
    communities = memberships.shape[-1]
    counts = torch.zeros((communities, len(catalog)))
    held_out = []
    for user_id, events in prefix.train_sequences.items():
        local = [event.item_id for event in events if event.domain_id == target_domain]
        if not local or memberships[target_domain, user_id].sum() == 0:
            continue
        community = int(memberships[target_domain, user_id].argmax())
        for item in local:
            counts[community, catalog_index[item]] += 1
        if (user_id, target_domain) in excluded:
            held_out.append((community, excluded[user_id, target_domain]))
    ranks = []
    for community, item in held_out:
        scores = counts[community]
        target = scores[catalog_index[item]]
        ranks.append(int((scores >= target).sum()))
    result = {"examples": len(ranks), "candidate_scope": "full_target_catalog",
              "definition": "last canonical-train event per user/domain excluded before graph fitting, clustering and popularity counts; no LLM",
              "strict_prefix_refit": True}
    for k in ks:
        result[f"recall@{k}"] = _mean(rank <= k for rank in ranks)
        result[f"ndcg@{k}"] = _mean(1 / math.log2(rank + 1) if rank <= k else 0 for rank in ranks)
    return result


def _profile_quality(cache_dir: Path, ks: Sequence[int]) -> dict[str, Any]:
    evidence = json.loads((cache_dir / "evidence.json").read_text(encoding="utf-8"))
    profiles = {}
    for line in (cache_dir / "profiles.jsonl").read_text(encoding="utf-8").splitlines():
        value = json.loads(line); profiles[value["community_id"]] = value
    interest_recalls = {k: [] for k in ks}; transition_hits, support = [], []
    reconstruction = []
    rows = []
    for row in evidence:
        profile = profiles.get(row["community_id"], {})
        profile_strings = _flatten_strings(profile)
        profile_tokens = _tokens(profile_strings)
        interest = row["evidence"]["interest_categories"]
        transitions = row["evidence"]["sequential_transitions"]
        interest_tokens = [_tokens(value["text"]) for value in interest]
        transition_tokens = [_tokens(value["text"]) for value in transitions]
        for k in ks:
            chosen = interest_tokens[:k]
            interest_recalls[k].append(_mean(bool(tokens & profile_tokens) for tokens in chosen))
        transition_hits.append(_mean(bool(tokens & profile_tokens) for tokens in transition_tokens))
        evidence_tokens = _tokens([value["text"] for group in row["evidence"].values()
                                   for value in group])
        phrase_support = [bool(_tokens(text) & evidence_tokens) for text in profile_strings
                          if _tokens(text)]
        support.append(_mean(phrase_support))
        weighted = interest + transitions
        reconstruction.append(sum(float(value["weight"]) for value in weighted
                                  if _tokens(value["text"]) & profile_tokens))
        rows.append({"community_id": row["community_id"], "evidence_support": support[-1],
                     "train_behavior_reconstruction": reconstruction[-1]})
    return {
        "communities": len(rows),
        "train_behavior_reconstruction": _mean(reconstruction),
        "interest_prototype_recall": {f"recall@{k}": _mean(values)
                                      for k, values in interest_recalls.items()},
        "behavior_transition_prediction_accuracy": _mean(transition_hits),
        "llm_evidence_support_rate": _mean(support),
        "support_check_definition": "lexical evidence coverage; conservative audit proxy, not an entailment model",
        "per_community": rows,
    }


def _evaluate_variants(model: CommCDSRModel, evaluator, test_loader, device) -> dict[str, Any]:
    variants = {
        "community_transfer": (False, None, False),
        "target_only": (True, None, False),
        "semantic_bridge_only": (False, 0, False),
        "behavior_bridge_only": (False, 1, False),
        "structure_bridge_only": (False, 2, False),
        "without_llm_profiles": (False, None, True),
    }
    results = {}
    try:
        for name, (disable_transfer, scheme, disable_profiles) in variants.items():
            model.audit_disable_transfer = disable_transfer
            model.audit_bridge_scheme = scheme
            model.audit_disable_profiles = disable_profiles
            results[name] = evaluator.evaluate(model, test_loader, device=device, split="test")
    finally:
        model.audit_disable_transfer = False
        model.audit_bridge_scheme = None
        model.audit_disable_profiles = False
    baseline = {row["user_id"]: row for row in results["target_only"].get("per_user", [])}
    transfer = {row["user_id"]: row for row in results["community_transfer"].get("per_user", [])}
    differences = []
    for user_id in sorted(baseline.keys() & transfer.keys()):
        delta = float(transfer[user_id]["ndcg@10"] - baseline[user_id]["ndcg@10"])
        differences.append({"user_id": user_id, "target_only_ndcg@10": baseline[user_id]["ndcg@10"],
                            "community_transfer_ndcg@10": transfer[user_id]["ndcg@10"], "delta": delta})
    positive = sum(row["delta"] > 1e-12 for row in differences)
    negative = sum(row["delta"] < -1e-12 for row in differences)
    paired = {
        "users": len(differences), "positive_transfer_rate": positive / max(len(differences), 1),
        "negative_transfer_rate": negative / max(len(differences), 1),
        "unchanged_rate": (len(differences) - positive - negative) / max(len(differences), 1),
        "mean_ndcg@10_delta": _mean(row["delta"] for row in differences),
        "per_user": differences,
    }
    compact = {name: value["overall"] for name, value in results.items()}
    compact["bridge_gain_over_target_only"] = {
        name: {metric: scores.get(metric, 0.0) - compact["target_only"].get(metric, 0.0)
               for metric in ("recall@10", "ndcg@10")}
        for name, scores in compact.items() if name not in {"target_only", "bridge_gain_over_target_only"}
    }
    return {"variants": compact, "paired_user_transfer": paired,
            "ablation_note": "checkpoint inference interventions, not retrained ablations; target_only uses identical normalization; without_llm_profiles removes memory values only and preserves keys, routing and fixed semantic OT"}


def _bridge_diagnostics(model: CommCDSRModel, threshold: float) -> dict[str, Any]:
    names = ("semantic", "behavior", "structure")
    rows = []
    target = model.target_domain_id
    for source in range(model.num_domains):
        if source == target:
            continue
        for scheme, name in enumerate(names):
            plan = model.base_bridges[target, source, scheme].detach().cpu()
            target_mask = model.community_mask[target].cpu()
            source_mask = model.community_mask[source].cpu()
            plan = plan[target_mask][:, source_mask]
            conditional = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
            entropy = -(conditional.clamp_min(1e-12) * conditional.clamp_min(1e-12).log()).sum(1)
            entropy /= math.log(max(conditional.shape[1], 2))
            rows.append({"source_domain_id": source, "target_domain_id": target, "bridge": name,
                         "normalized_mapping_entropy": float(entropy.mean()),
                         "unmatched_threshold": threshold,
                         "unmatched_ratio": float(plan.sum(dim=1).lt(threshold).float().mean()),
                         "mean_unmatched_mass": float((1 - plan.sum(dim=1)).mean()),
                         "ot_reliability": float(
                             model.bridge_reliability[target, source, scheme][model.community_mask[target]].mean()
                         )})
    return {"mappings": rows, "identity_used_for_bridge": False,
            "identity_audit_policy": "overlap identity may be inspected after evaluation only"}


def _rerank_diagnostics(
    neural: Mapping[str, Any] | None,
    reranked: Mapping[str, Any] | None,
    runtime: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if neural is None or reranked is None:
        return {"enabled": False}
    before = {row["user_id"]: row for row in neural.get("per_user", [])}
    after = {row["user_id"]: row for row in reranked.get("per_user", [])}
    rows = []
    for user_id in sorted(before.keys() & after.keys()):
        delta = float(after[user_id]["ndcg@10"] - before[user_id]["ndcg@10"])
        rows.append({
            "user_id": user_id,
            "neural_ndcg@10": before[user_id]["ndcg@10"],
            "llm_reranked_ndcg@10": after[user_id]["ndcg@10"],
            "delta": delta,
        })
    positive = sum(row["delta"] > 1e-12 for row in rows)
    negative = sum(row["delta"] < -1e-12 for row in rows)
    metrics = ("recall@5", "ndcg@5", "recall@10", "ndcg@10",
               "recall@20", "ndcg@20")
    return {
        "enabled": True,
        "neural_overall": neural["overall"],
        "llm_reranked_overall": reranked["overall"],
        "overall_delta": {
            metric: float(reranked["overall"].get(metric, 0.0)
                          - neural["overall"].get(metric, 0.0))
            for metric in metrics
        },
        "paired_users": len(rows),
        "positive_rerank_rate": positive / max(len(rows), 1),
        "negative_rerank_rate": negative / max(len(rows), 1),
        "unchanged_rate": (len(rows) - positive - negative) / max(len(rows), 1),
        "mean_ndcg@10_delta": _mean(row["delta"] for row in rows),
        "runtime": dict(runtime or {}),
        "per_user": rows,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# CommCDSR 三模块质量验证", "",
             "本报告所有社区形成、画像和桥均只使用 canonical train；重叠用户身份未用于建桥。", "",
             "## 社区划分", ""]
    for domain in report["community_partition"]["domains"]:
        stability = ", ".join(f"seed={x['seed']}: ARI={x['ari']:.4f}, NMI={x['nmi']:.4f}"
                              for x in domain["seed_stability"])
        lines += [f"- {domain['domain']} 规模：`{domain['community_sizes']}`",
                  f"- 稳定性：{stability}",
                  f"- 域内/社区间余弦距离：{domain['mean_intra_cosine_distance']:.4f} / {domain['mean_inter_centroid_cosine_distance']:.4f}",
                  f"- 类别/行为转移 JS divergence：{domain['mean_pairwise_category_js_divergence']:.4f} / {domain['mean_pairwise_transition_js_divergence']:.4f}", ""]
    held = report["community_only_next_item"]
    lines += [f"社区归属预测 train-held-out：Recall@10={held['recall@10']:.4f}, NDCG@10={held['ndcg@10']:.4f}", "",
              "## 社区画像", "",
              f"- train 行为重构：{report['community_profile']['train_behavior_reconstruction']:.4f}",
              f"- 兴趣原型：`{report['community_profile']['interest_prototype_recall']}`",
              f"- 行为转移预测准确率：{report['community_profile']['behavior_transition_prediction_accuracy']:.4f}",
              f"- evidence 支持率：{report['community_profile']['llm_evidence_support_rate']:.4f}", "",
              "## 社区桥与迁移", ""]
    for row in report["community_bridge"]["mappings"]:
        lines.append(f"- {row['bridge']}：entropy={row['normalized_mapping_entropy']:.4f}, unmatched={row['unmatched_ratio']:.4f}, reliability={row['ot_reliability']:.4f}")
    paired = report["recommendation_audits"]["paired_user_transfer"]
    lines += ["", f"- 正迁移率：{paired['positive_transfer_rate']:.4f}",
              f"- 负迁移率：{paired['negative_transfer_rate']:.4f}",
              f"- 平均逐用户 NDCG@10 增益：{paired['mean_ndcg@10_delta']:.6f}", "",
              "## 选择性 LLM Top-K", ""]
    rerank = report.get("llm_rerank", {"enabled": False})
    if rerank.get("enabled"):
        lines += [
            f"- 神经/LLM NDCG@10：{rerank['neural_overall']['ndcg@10']:.4f} / {rerank['llm_reranked_overall']['ndcg@10']:.4f}",
            f"- LLM NDCG@10 增益：{rerank['overall_delta']['ndcg@10']:.6f}",
            f"- LLM 正向/负向重排率：{rerank['positive_rerank_rate']:.4f} / {rerank['negative_rerank_rate']:.4f}",
            f"- 请求统计：`{rerank['runtime'].get('stats', {})}`", "",
        ]
    else:
        lines += ["- 本次未启用 LLM Top-K。", ""]
    lines += ["完整分桥结果、消融结果及逐用户差异见同目录 JSON。", ""]
    return "\n".join(lines)


def validate_comm_cdsr(
    *, model: CommCDSRModel, corpus: CanonicalCorpus, features: CommCDSRFeatures,
    domain_entries: Sequence[Mapping[str, Any]], settings: Mapping[str, Any],
    evaluator, test_loader, device, output_dir: str | Path, seed: int,
    logger=None, neural_test_result: Mapping[str, Any] | None = None,
    reranked_test_result: Mapping[str, Any] | None = None,
    rerank_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    validation = dict(settings)
    seeds = list(map(int, validation.get("stability_seeds", [seed, seed + 1, seed + 2])))
    ks = tuple(map(int, validation.get("prototype_topk", [5, 10])))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": corpus.metadata.get("dataset_name", "unknown"),
        "data_policy": "canonical_train_only_for_formation_profiles_and_bridges",
        "community_partition": _community_intrinsics(
            corpus, features, domain_entries, validation["community_memory"], seeds
        ),
        "community_only_next_item": _community_only_next_item(
            corpus, features, model.target_domain_id, ks,
            {**validation["community_memory"], "audit_seed": seed},
        ),
        "community_profile": _profile_quality(features.cache_dir, ks),
        "community_bridge": _bridge_diagnostics(
            model, float(validation.get("unmatched_probability_threshold", 0.1))
        ),
        "recommendation_audits": _evaluate_variants(model, evaluator, test_loader, device),
        "llm_rerank": _rerank_diagnostics(
            neural_test_result, reranked_test_result, rerank_report
        ),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = output / f"comm_cdsr_validation_{stamp}.json"
    markdown_file = output / f"comm_cdsr_validation_{stamp}.md"
    json_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_file.write_text(_markdown(report), encoding="utf-8")
    if logger:
        logger.info("CommCDSR validation reports: %s, %s", json_file, markdown_file)
    return {"json": str(json_file), "markdown": str(markdown_file), "summary": {
        "positive_transfer_rate": report["recommendation_audits"]["paired_user_transfer"]["positive_transfer_rate"],
        "negative_transfer_rate": report["recommendation_audits"]["paired_user_transfer"]["negative_transfer_rate"],
    }}
