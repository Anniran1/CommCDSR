"""Train-only community formation and LLM profiles for CommCDSR.

This module implements community formation and bridge preparation from
``method.txt``. Review sidecars may contain every retained interaction, but
only canonical-training ``(user, item)`` pairs are permitted as model input.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests
import torch
import torch.nn.functional as F

from cdsr.data.schema import CanonicalCorpus
from cdsr.features.usage import TokenUsageTracker


PROFILE_VERSION = "comm_cdsr_v8_conditional_memory"
SCHEMA_VERSION = "comm_cdsr.profile.v4"


@dataclass(frozen=True)
class CommCDSRFeatures:
    memberships: torch.Tensor
    coarse_memberships: torch.Tensor
    fine_to_coarse: torch.Tensor
    item_fine_memberships: torch.Tensor
    fine_graph_profiles: torch.Tensor
    fine_shared_profiles: torch.Tensor
    fine_private_profiles: torch.Tensor
    source: str
    cache_dir: Path | None = None
    manifest: Mapping[str, Any] = field(default_factory=dict)
    api_usage: Mapping[str, Mapping[str, int | str]] = field(default_factory=dict)
    api_usage_cumulative: Mapping[str, Mapping[str, int | str]] = field(default_factory=dict)
    memory_keys: torch.Tensor | None = None
    memory_values: torch.Tensor | None = None
    memory_support: torch.Tensor | None = None
    concept_behavior: torch.Tensor | None = None
    structure_profiles: torch.Tensor | None = None

    @property
    def interest_profiles(self) -> torch.Tensor:
        """Backward-compatible alias for the transferable fine memory."""
        return self.fine_shared_profiles

    @property
    def behavior_profiles(self) -> torch.Tensor:
        """Backward-compatible alias for the domain-private fine memory."""
        return self.fine_private_profiles


def resolve_num_communities(
    corpus: CanonicalCorpus, requested: Any, settings: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    active = [
        sum(any(event.domain_id == domain for event in events)
            for events in corpus.train_sequences.values())
        for domain in range(corpus.num_domains)
    ]
    if str(requested).strip().lower() == "auto":
        target = max(1, int(settings.get("target_users_per_community", 256)))
        minimum = max(2, int(settings.get("min_communities", 8)))
        maximum = max(minimum, int(settings.get("max_communities", 32)))
        count = max(minimum, min(maximum, math.ceil(max(active, default=0) / target)))
        if max(active, default=0):
            count = min(count, max(active))
        mode = "auto"
    else:
        count = int(requested)
        if count < 2:
            raise ValueError("num_communities must be 'auto' or an integer >= 2")
        target, minimum, maximum, mode = None, count, count, "fixed"
    coarse = int(settings.get("coarse_communities", min(8, count)))
    if count > 2 and coarse >= count:
        coarse = max(2, count // 2)
    coarse = max(2, min(coarse, count))
    return count, {
        "mode": mode,
        "active_train_users_by_domain": active,
        "target_users_per_community": target,
        "min_communities": minimum,
        "max_communities": maximum,
        "resolved_communities": count,
        "resolved_coarse_communities": coarse,
        "estimated_users_per_community": [round(value / count, 3) for value in active],
        "planned_chat_requests": corpus.num_domains * (count + coarse),
    }


def _bucket(text: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % size


def _safe_text(value: Any, limit: int = 240) -> str:
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()[:limit]


def _read_items(path: str | Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        names = [part.split(":", 1)[0] for part in next(stream).rstrip("\n").split("\t")]
        for line in stream:
            values = line.rstrip("\n").split("\t")
            values += [""] * (len(names) - len(values))
            if values and values[0]:
                rows[values[0]] = dict(zip(names, values))
    return rows


def _item_metadata(
    corpus: CanonicalCorpus, domain_entries: Sequence[Mapping[str, Any]]
) -> dict[int, dict[str, str]]:
    reverse = {mapped: raw for raw, mapped in corpus.global_item_to_id.items()}
    output: dict[int, dict[str, str]] = {}
    for domain, entry in enumerate(domain_entries):
        rows = _read_items(entry["item_file"])
        review_texts: defaultdict[int, list[str]] = defaultdict(list)
        review_file = entry.get("review_file")
        if review_file:
            review_path = Path(review_file)
            if not review_path.is_file():
                raise FileNotFoundError(
                    f"Configured review_file does not exist: {review_path}"
                )
            # This allow-list is deliberately derived after the canonical
            # leave-one-out split. Never replace it with all raw interactions:
            # doing so would expose validation/test review text to the LLM.
            allowed = {
                (user_id, event.item_id)
                for user_id, events in corpus.train_sequences.items()
                for event in events if event.domain_id == domain
            }
            opener = gzip.open if str(review_path).endswith(".gz") else open
            with opener(review_path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        review = json.loads(line)
                        user_id = corpus.user_to_id.get(str(review.get("user_id", "")))
                        raw_item = str(review.get("item_id") or review.get("asin") or "")
                        item_id = corpus.global_item_to_id.get(
                            f"{entry['name']}␟{raw_item}"
                        )
                    except (ValueError, TypeError):
                        continue
                    if (user_id, item_id) not in allowed or len(review_texts[item_id]) >= 2:
                        continue
                    text = _safe_text(review.get("text"), 240)
                    if text:
                        review_texts[item_id].append(text)
        for item_id in corpus.domain_catalogs[domain]:
            raw_id = reverse[item_id].split("␟", 1)[-1]
            row = rows.get(raw_id, {})
            output[item_id] = {
                "title": _safe_text(row.get("item_title") or row.get("title") or raw_id, 120),
                "category": _safe_text(row.get("categories") or row.get("category") or "Unknown", 160),
                "brand": _safe_text(row.get("brand"), 120),
                "content": _safe_text(
                    " ".join(review_texts[item_id])
                    or row.get("review") or row.get("description"), 320
                ),
                "domain": str(entry["name"]),
            }
    return output


def _soft_cluster(
    features: torch.Tensor, communities: int, *, seed: int, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic fuzzy clustering used only to bootstrap community nodes."""

    if not len(features):
        return torch.empty((0, communities)), torch.zeros((communities, features.shape[-1]))
    count = min(communities, len(features))
    generator = torch.Generator().manual_seed(seed)
    selected = [int(torch.randint(len(features), (1,), generator=generator))]
    while len(selected) < count:
        distance = 1.0 - (features @ features[selected].T).max(dim=1).values
        distance[selected] = -1
        selected.append(int(distance.argmax()))
    centroids = features[selected]
    for _ in range(20):
        membership = torch.softmax((features @ centroids.T) / temperature, dim=1)
        centroids = F.normalize(membership.T @ features, dim=1)
    membership = torch.softmax((features @ centroids.T) / temperature, dim=1)
    if count < communities:
        membership = F.pad(membership, (0, communities - count))
        centroids = F.pad(centroids, (0, 0, 0, communities - count))
    return membership, centroids


def _hard_spherical_cluster(
    features: torch.Tensor,
    communities: int,
    *,
    seed: int,
    iterations: int = 50,
    min_cluster_fraction: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic, non-collapsing spherical k-means.

    Farthest-first initialization makes the result deterministic for a fixed
    corpus and seed. A small lower-bound repair prevents empty/tiny clusters.
    Every active user belongs to exactly one community.
    """

    if not len(features):
        return torch.empty((0, communities)), torch.zeros((communities, features.shape[-1]))
    if communities > len(features):
        raise ValueError(
            f"hard community count {communities} exceeds active users {len(features)}"
        )
    if not 0.0 <= min_cluster_fraction < 1.0:
        raise ValueError("min_cluster_fraction must be in [0,1)")
    values = F.normalize(features.float(), dim=1)
    generator = torch.Generator().manual_seed(seed)
    selected = [int(torch.randint(len(values), (1,), generator=generator))]
    while len(selected) < communities:
        distance = 1.0 - (values @ values[selected].T).max(dim=1).values
        distance[selected] = -1.0
        selected.append(int(distance.argmax()))
    centroids = values[selected].clone()
    minimum = max(1, int(math.ceil(len(values) * min_cluster_fraction)))
    minimum = min(minimum, len(values) // communities)
    assignments = torch.full((len(values),), -1, dtype=torch.long)
    for _ in range(max(1, iterations)):
        similarity = values @ centroids.T
        updated = similarity.argmax(dim=1)
        counts = torch.bincount(updated, minlength=communities)
        for target in range(communities):
            while int(counts[target]) < minimum:
                candidates = (counts[updated] > minimum).nonzero(as_tuple=False).flatten()
                if not len(candidates):
                    raise RuntimeError("unable to satisfy hard-community minimum size")
                current = updated[candidates]
                penalty = similarity[candidates, current] - similarity[candidates, target]
                moved = int(candidates[penalty.argmin()])
                old = int(updated[moved])
                updated[moved] = target
                counts[old] -= 1
                counts[target] += 1
        centroids = torch.stack([
            F.normalize(values[updated.eq(community)].mean(dim=0), dim=0)
            for community in range(communities)
        ])
        if torch.equal(updated, assignments):
            break
        assignments = updated.clone()
    membership = F.one_hot(assignments, num_classes=communities).float()
    return membership, centroids


def _cluster(
    features: torch.Tensor,
    communities: int,
    *,
    seed: int,
    temperature: float,
    assignment: str,
    iterations: int,
    min_cluster_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if assignment == "soft":
        return _soft_cluster(
            features, communities, seed=seed, temperature=temperature,
        )
    if assignment == "hard_spherical_kmeans":
        return _hard_spherical_cluster(
            features, communities, seed=seed, iterations=iterations,
            min_cluster_fraction=min_cluster_fraction,
        )
    raise ValueError(
        "community_assignment must be soft or hard_spherical_kmeans"
    )


def _dual_view_features(
    corpus: CanonicalCorpus,
    domain: int,
    communities: int,
    *,
    seed: int,
    feature_dim: int,
    temperature: float,
    assignment: str = "soft",
    clustering_iterations: int = 50,
    min_cluster_fraction: float = 0.01,
) -> tuple[list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Explicit collaborative and directed-transition views plus fuzzy nodes."""

    users = sorted(
        user_id for user_id, events in corpus.train_sequences.items()
        if any(event.domain_id == domain for event in events)
    )
    if not users:
        empty = torch.empty((0, feature_dim))
        return [], empty, empty, torch.empty((0, communities)), torch.zeros((communities, feature_dim))
    feature_dim = max(16, feature_dim)
    sequences = {
        user_id: [event.item_id for event in corpus.train_sequences[user_id]
                  if event.domain_id == domain]
        for user_id in users
    }
    item_degrees = Counter(item for values in sequences.values() for item in set(values))
    transition_edges = Counter(
        (left, right) for values in sequences.values() for left, right in zip(values, values[1:])
    )
    out_degrees = Counter()
    in_degrees = Counter()
    for (left, right), count in transition_edges.items():
        out_degrees[left] += count
        in_degrees[right] += count

    def item_seed(item_id: int) -> torch.Tensor:
        value = torch.zeros(feature_dim - 3)
        for view in range(4):
            bucket = _bucket(f"item-node:{view}:{item_id}", feature_dim - 3)
            sign = 1.0 if _bucket(f"item-sign:{view}:{item_id}", 2) else -1.0
            value[bucket] += sign
        return F.normalize(value, dim=0)

    item_ids = sorted({item for values in sequences.values() for item in values})
    seeds = {item: item_seed(item) for item in item_ids}
    transition_items = {item: seeds[item].clone() for item in item_ids}
    for (left, right), count in transition_edges.items():
        weight = count / math.sqrt(max(out_degrees[left] * in_degrees[right], 1))
        transition_items[left] += weight * torch.roll(seeds[right], shifts=1)
        transition_items[right] += weight * torch.roll(seeds[left], shifts=2)
    transition_items = {
        item: F.normalize(value, dim=0) for item, value in transition_items.items()
    }
    collaborative = torch.zeros((len(users), feature_dim), dtype=torch.float32)
    transition = torch.zeros((len(users), feature_dim), dtype=torch.float32)
    for row, user_id in enumerate(users):
        items = sequences[user_id]
        user_degree = max(len(set(items)), 1)
        for item_id, count in Counter(items).items():
            # One normalized LightGCN-style item -> user propagation step.
            weight = math.log1p(count) / math.sqrt(user_degree * item_degrees[item_id])
            collaborative[row, :-3] += weight * seeds[item_id]
        for position, item_id in enumerate(items):
            # Directed transition-graph item states, with recent events weighted more.
            recency = 1.0 / math.sqrt(len(items) - position)
            transition[row, :-3] += recency * transition_items[item_id]
        collaborative[row, -3] = math.log1p(len(items))
        collaborative[row, -2] = len(set(items)) / max(len(items), 1)
        collaborative[row, -1] = 1.0
        transition[row, -3] = math.log1p(max(0, len(items) - 1))
        transition[row, -2] = 1.0 / math.sqrt(max(len(items), 1))
        transition[row, -1] = 1.0
    def normalize_view(values: torch.Tensor) -> torch.Tensor:
        # The raw log-length statistics can be an order of magnitude larger
        # than the propagated item vector and previously made every user look
        # almost identical. Keep behavior statistics, but z-normalize and
        # down-weight them so item preference/transition structure dominates.
        structural = F.normalize(values[:, :-3], dim=1)
        statistics = values[:, -3:-1]
        statistics = (
            statistics - statistics.mean(dim=0, keepdim=True)
        ) / statistics.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return F.normalize(torch.cat((structural, 0.25 * statistics,
                                      torch.zeros_like(values[:, -1:])), dim=1), dim=1)

    collaborative = normalize_view(collaborative)
    transition = normalize_view(transition)
    # A neutral gate bootstraps the hierarchy; the model learns the gate from
    # recommendation and community-consistency losses.
    fused = F.normalize(0.5 * collaborative + 0.5 * transition, dim=1)
    membership, centroids = _cluster(
        fused, communities, seed=seed + 1009 * domain, temperature=temperature,
        assignment=assignment, iterations=clustering_iterations,
        min_cluster_fraction=min_cluster_fraction,
    )
    return users, collaborative, transition, membership, centroids


def build_comm_cdsr_hierarchy(
    corpus: CanonicalCorpus, fine_communities: int, coarse_communities: int, *,
    seed: int, feature_dim: int = 256, temperature: float = 0.2,
    assignment: str = "soft", clustering_iterations: int = 50,
    min_cluster_fraction: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Bootstrap train-only dual-view fine/coarse community hierarchy."""

    if coarse_communities < 2 or coarse_communities > fine_communities:
        raise ValueError("coarse communities must be in [2, fine communities]")
    shape = (corpus.num_domains, corpus.num_users + 1, feature_dim)
    collaborative = torch.zeros(shape, dtype=torch.float32)
    transition = torch.zeros(shape, dtype=torch.float32)
    memberships = torch.zeros(
        (corpus.num_domains, corpus.num_users + 1, fine_communities), dtype=torch.float32
    )
    fine_graph = torch.zeros(
        (corpus.num_domains, fine_communities, feature_dim), dtype=torch.float32
    )
    hierarchy = torch.zeros(
        (corpus.num_domains, fine_communities, coarse_communities), dtype=torch.float32
    )
    for domain in range(corpus.num_domains):
        users, collab, trans, values, centroids = _dual_view_features(
            corpus, domain, fine_communities, seed=seed,
            feature_dim=feature_dim, temperature=temperature,
            assignment=assignment, clustering_iterations=clustering_iterations,
            min_cluster_fraction=min_cluster_fraction,
        )
        if not users:
            continue
        index = torch.tensor(users)
        collaborative[domain, index] = collab
        transition[domain, index] = trans
        memberships[domain, index] = values
        fine_graph[domain] = centroids
        active = centroids.norm(dim=1).gt(0)
        relation, coarse = _cluster(
            centroids[active], coarse_communities,
            seed=seed + 7919 * (domain + 1), temperature=temperature,
            assignment=assignment, iterations=clustering_iterations,
            min_cluster_fraction=0.0,
        )
        hierarchy[domain, active] = relation
    coarse_memberships = torch.einsum("duk,dkc->duc", memberships, hierarchy)
    return {
        "memberships": memberships,
        "coarse_memberships": coarse_memberships,
        "fine_to_coarse": hierarchy,
        "fine_graph_profiles": fine_graph,
        "collaborative_user_features": collaborative,
        "transition_user_features": transition,
    }


def _light_gnn_domain(
    corpus: CanonicalCorpus, domain: int, *, seed: int, dimension: int,
    epochs: int, layers: int,
) -> tuple[list[int], list[int], torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, list[tuple[int, int, int]], Counter[tuple[int, int]]]:
    """Encode collaborative and sequential-transition views independently.

    The collaborative view is a conventional user--item LightGCN.  The
    transition view is a directed item GNN over consecutive train events;
    users obtain a recency-weighted pooling of their transition-item states.
    No community objective is mixed into either encoder.
    """

    users = sorted(user_id for user_id, events in corpus.train_sequences.items()
                   if any(event.domain_id == domain for event in events))
    sequences = {user_id: [event.item_id for event in corpus.train_sequences[user_id]
                           if event.domain_id == domain] for user_id in users}
    items = sorted({item for sequence in sequences.values() for item in sequence})
    user_index = {user_id: row for row, user_id in enumerate(users)}
    item_index = {item_id: row for row, item_id in enumerate(items)}
    counts = Counter((user_index[user_id], item_index[item])
                     for user_id, sequence in sequences.items() for item in sequence)
    edges = [(user, item, count) for (user, item), count in sorted(counts.items())]
    transitions = Counter(
        (item_index[left], item_index[right])
        for sequence in sequences.values() for left, right in zip(sequence, sequence[1:])
        if left != right
    )
    generator = torch.Generator().manual_seed(seed + 104729 * (domain + 1))
    collaborative_user = torch.nn.Parameter(
        torch.randn(len(users), dimension, generator=generator) * 0.05
    )
    collaborative_item = torch.nn.Parameter(
        torch.randn(len(items), dimension, generator=generator) * 0.05
    )
    transition_item = torch.nn.Parameter(
        torch.randn(len(items), dimension, generator=generator) * 0.05
    )
    # Separate predecessor/successor coordinates make the score directional.
    transition_out_projection = torch.nn.Parameter(
        torch.eye(dimension) + torch.randn(dimension, dimension, generator=generator) * 0.01
    )
    transition_in_projection = torch.nn.Parameter(
        torch.eye(dimension) + torch.randn(dimension, dimension, generator=generator) * 0.01
    )
    optimizer = torch.optim.Adam(
        (collaborative_user, collaborative_item, transition_item,
         transition_out_projection, transition_in_projection), lr=0.02
    )
    edge_user = torch.tensor([edge[0] for edge in edges], dtype=torch.long)
    edge_item = torch.tensor([edge[1] for edge in edges], dtype=torch.long)
    edge_weight = torch.tensor([math.log1p(edge[2]) for edge in edges], dtype=torch.float32)
    user_degree = torch.zeros(len(users)).index_add_(0, edge_user, edge_weight)
    item_degree = torch.zeros(len(items)).index_add_(0, edge_item, edge_weight)
    norm = edge_weight / torch.sqrt(
        user_degree[edge_user].clamp_min(1e-8) * item_degree[edge_item].clamp_min(1e-8)
    )
    transition_left = torch.tensor([edge[0] for edge in transitions], dtype=torch.long)
    transition_right = torch.tensor([edge[1] for edge in transitions], dtype=torch.long)
    transition_weight = torch.tensor(
        [math.log1p(value) for value in transitions.values()], dtype=torch.float32
    )
    transition_out = torch.zeros(len(items)).index_add_(
        0, transition_left, transition_weight
    ) if len(transitions) else torch.zeros(len(items))
    transition_in = torch.zeros(len(items)).index_add_(
        0, transition_right, transition_weight
    ) if len(transitions) else torch.zeros(len(items))
    transition_norm = (
        transition_weight / torch.sqrt(
            transition_out[transition_left].clamp_min(1e-8)
            * transition_in[transition_right].clamp_min(1e-8)
        )
        if len(transitions) else transition_weight
    )

    def collaborative_propagate() -> tuple[torch.Tensor, torch.Tensor]:
        current_user, current_item = collaborative_user, collaborative_item
        users_sum, items_sum = current_user, current_item
        for _ in range(max(1, layers)):
            next_user = torch.zeros_like(current_user).index_add_(
                0, edge_user, current_item[edge_item] * norm.unsqueeze(1)
            )
            next_item = torch.zeros_like(current_item).index_add_(
                0, edge_item, current_user[edge_user] * norm.unsqueeze(1)
            )
            current_user, current_item = next_user, next_item
            users_sum, items_sum = users_sum + current_user, items_sum + current_item
        scale = float(max(1, layers) + 1)
        return users_sum / scale, items_sum / scale

    def transition_propagate() -> torch.Tensor:
        current, accumulated = transition_item, transition_item
        for _ in range(max(1, layers)):
            propagated = torch.zeros_like(current)
            if len(transition_left):
                propagated.index_add_(
                    0, transition_right,
                    current[transition_left] * transition_norm.unsqueeze(1),
                )
            current = propagated
            accumulated = accumulated + current
        return accumulated / float(max(1, layers) + 1)

    for _ in range(max(1, epochs)):
        learned_users, learned_items = collaborative_propagate()
        learned_transition_items = transition_propagate()
        negative = torch.randint(len(items), edge_item.shape, generator=generator)
        positive_score = (learned_users[edge_user] * learned_items[edge_item]).sum(dim=1)
        negative_score = (learned_users[edge_user] * learned_items[negative]).sum(dim=1)
        loss = F.softplus(negative_score - positive_score).mean()
        if len(transition_left):
            transition_negative = torch.randint(
                len(items), transition_right.shape, generator=generator
            )
            transition_positive = (
                (learned_transition_items[transition_left] @ transition_out_projection)
                * (learned_transition_items[transition_right] @ transition_in_projection)
            ).sum(dim=1)
            transition_wrong = (
                (learned_transition_items[transition_left] @ transition_out_projection)
                * (learned_transition_items[transition_negative] @ transition_in_projection)
            ).sum(dim=1)
            loss = loss + F.softplus(transition_wrong - transition_positive).mean()
        loss = loss + 1e-4 * (
            collaborative_user.square().mean()
            + collaborative_item.square().mean()
            + transition_item.square().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        learned_users, learned_items = collaborative_propagate()
        learned_transition_items = transition_propagate()
        transition_users = torch.zeros_like(learned_users)
        for user_id, sequence in sequences.items():
            row = user_index[user_id]
            indexed = [item_index[item] for item in sequence]
            # Recent transitions carry more current-intent information while
            # remaining a train-only summary.
            weights = torch.linspace(1.0, 2.0, len(indexed))
            transition_users[row] = (
                learned_transition_items[indexed] * weights.unsqueeze(1)
            ).sum(dim=0) / weights.sum()
        collaborative_users = F.normalize(learned_users, dim=1)
        transition_users = F.normalize(transition_users, dim=1)
        # Concatenation preserves independent coordinate systems. Its dot
        # product is the mean of the two view similarities; no latent addition.
        fused_users = F.normalize(
            torch.cat((collaborative_users, transition_users), dim=1), dim=1,
        )
        fused_items = F.normalize(
            torch.cat((F.normalize(learned_items, dim=1),
                       F.normalize(learned_transition_items, dim=1)), dim=1),
            dim=1,
        )
        return (users, items, collaborative_users.cpu(), transition_users.cpu(),
                fused_users.cpu(), fused_items.cpu(), edges, transitions)


def _merge_small_communities(labels: torch.Tensor, states: torch.Tensor,
                             minimum: int) -> torch.Tensor:
    """Merge statistically fragile Louvain fragments without prescribing K."""

    labels = labels.clone()
    # A minimum larger than the population is the degenerate "no partition"
    # ablation: everyone in one community (the pairwise loop below cannot
    # collapse below two groups by design).
    if minimum >= len(labels):
        return torch.zeros(len(labels), dtype=torch.long)
    while True:
        unique, counts = labels.unique(return_counts=True)
        small = unique[counts < minimum]
        if not len(small) or len(unique) <= 2:
            break
        community = int(small[counts[counts < minimum].argmin()])
        source = F.normalize(states[labels.eq(community)].mean(dim=0), dim=0)
        candidates = unique[unique.ne(community)]
        centroids = torch.stack([
            F.normalize(states[labels.eq(int(value))].mean(dim=0), dim=0)
            for value in candidates
        ])
        destination = int(candidates[(centroids @ source).argmax()])
        labels[labels.eq(community)] = destination
    remap = {int(value): index for index, value in enumerate(labels.unique(sorted=True))}
    return torch.tensor([remap[int(value)] for value in labels], dtype=torch.long)




def training_prefix_corpus(corpus: CanonicalCorpus) -> tuple[CanonicalCorpus, dict]:
    """Remove the last canonical-train event per user/domain, before ANY fitting."""
    prefixes, heldout = {}, {}
    for user, events in corpus.train_sequences.items():
        excluded = set()
        for domain in range(corpus.num_domains):
            indices = [i for i, event in enumerate(events) if event.domain_id == domain]
            if len(indices) >= 2:
                excluded.add(indices[-1])
                heldout[user, domain] = events[indices[-1]].item_id
        prefixes[user] = [event for i, event in enumerate(events) if i not in excluded]
    return replace(corpus, train_sequences=prefixes), heldout


def _user_graph(states: torch.Tensor):
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    neighbours = min(12, max(0, len(states) - 1))
    for start in range(0, len(states), 512):
        similarity = states[start:start + 512] @ states.T
        rows = torch.arange(start, min(start + 512, len(states)))
        similarity[torch.arange(len(rows)), rows] = -1
        scores, indices = similarity.topk(neighbours, dim=1)
        for offset, (values, targets) in enumerate(zip(scores.tolist(), indices.tolist())):
            for weight, right in zip(values, targets):
                if weight > 0:
                    left = start + offset
                    old = graph.get_edge_data(left, right, {}).get("weight", 0.0)
                    graph.add_edge(left, right, weight=max(old, weight))
    return graph


def _select_resolution(prefix, heldout, users, graph, domain, center, seed):
    """Small train-internal search. No validation/test or LLM calls involved."""
    import networkx as nx
    if not graph.number_of_edges():
        return center, []
    generator = random.Random(seed)
    perturbed = graph.copy()
    perturbed.remove_edges_from([edge for edge in graph.edges if generator.random() < 0.1])
    candidates = []
    for resolution in (center * 0.75, center, center * 1.5):
        groups = nx.community.louvain_communities(graph, resolution=resolution, seed=seed)
        alternate = (nx.community.louvain_communities(perturbed, resolution=resolution, seed=seed)
                     if perturbed.number_of_edges() else [{node} for node in graph.nodes])
        stability = sum(len(group) * max(len(group & other) / len(group | other)
                                       for other in alternate) for group in groups) / max(len(users), 1)
        gains = []
        for group in groups:
            counts = Counter(event.item_id for row in group
                             for event in prefix.train_sequences[users[row]]
                             if event.domain_id == domain)
            for row in group:
                item = heldout.get((users[row], domain))
                if item is None:
                    continue
                rank = (sum(value >= counts[item] for value in counts.values())
                        if counts[item] else len(prefix.domain_catalogs[domain]))
                gains.append(1 / math.log2(rank + 1) if rank <= 10 else 0.0)
        ndcg = sum(gains) / max(len(gains), 1)
        imbalance = max(map(len, groups)) / max(len(users), 1)
        candidates.append({"resolution": resolution, "train_prefix_ndcg@10": ndcg,
                           "edge_dropout_stability": stability, "largest_fraction": imbalance,
                           "score": ndcg + 0.01 * stability - 0.01 * imbalance})
    return max(candidates, key=lambda value: value["score"])["resolution"], candidates


def build_gnn_modularity_hierarchy(
    corpus: CanonicalCorpus, *, seed: int, feature_dim: int = 128,
    gnn_epochs: int = 12, gnn_layers: int = 2, modularity_resolution: float = 1.0,
    minimum_community_size: int | None = None,
    select_resolution: bool = True,
) -> dict[str, torch.Tensor | Any]:
    """Dual-view GNN states followed by simple Louvain communities.

    Louvain runs on a sparse user graph induced by the fused collaborative and
    directed-transition GNN representations. It determines the fine-community
    count itself; no desired number of communities or community-specific loss
    is used.
    """

    import networkx as nx

    learned = []
    selection_diagnostics = []
    prefix, heldout = training_prefix_corpus(corpus)
    for domain in range(corpus.num_domains):
        resolution, candidates = modularity_resolution, []
        if select_resolution:
            prefix_data = _light_gnn_domain(
                prefix, domain, seed=seed, dimension=feature_dim,
                epochs=gnn_epochs, layers=gnn_layers,
            )
            resolution, candidates = _select_resolution(
                prefix, heldout, prefix_data[0], _user_graph(prefix_data[4]),
                domain, modularity_resolution, seed + 1009 * domain,
            )
        (users, items, collaborative_states, transition_states, user_states,
         item_states, edges, transitions) = _light_gnn_domain(
            corpus, domain, seed=seed, dimension=feature_dim,
            epochs=gnn_epochs, layers=gnn_layers,
        )
        graph = _user_graph(user_states)
        modules = nx.community.louvain_communities(
            graph, weight="weight", resolution=resolution,
            seed=seed + 1009 * domain,
        ) if graph.number_of_edges() else [{node} for node in graph.nodes]
        modularity = nx.community.modularity(
            graph, modules, weight="weight", resolution=modularity_resolution
        ) if graph.number_of_edges() else 0.0
        labels = torch.full((len(users),), -1, dtype=torch.long)
        for community, nodes in enumerate(modules):
            for node in nodes:
                labels[int(node)] = community
        if bool(labels.lt(0).any()):
            raise RuntimeError("GNN modularity discovery left users unassigned")
        raw_count = int(labels.unique().numel())
        labels = _merge_small_communities(
            labels, user_states,
            minimum_community_size or max(8, int(math.ceil(math.log2(max(len(users), 2))))),
        )
        selection_diagnostics.append({
            "domain_id": domain,
            "selected_resolution": resolution,
            "resolution_candidates": candidates,
            "louvain_modularity": float(modularity),
            "raw_modules": raw_count,
            "modules_after_small_merge": int(labels.max()) + 1,
            "active_users": len(users),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "encoder": "collaborative_lightgcn+directed_transition_gnn",
            "fusion": "mean_of_independent_view_similarities",
        })
        learned.append((users, collaborative_states, transition_states,
                        user_states, labels, graph))

    fine_count = max(int(values[4].max()) + 1 for values in learned)
    # Zero padding is storage only: never split a domain to match another K.
    domain_labels = [values[4] for values in learned]
    fine_profiles = []
    for (_, _, _, states, _, _), labels in zip(learned, domain_labels):
        profiles = torch.stack([
            F.normalize(states[labels.eq(community)].mean(dim=0), dim=0)
            for community in range(int(labels.max()) + 1)
        ])
        fine_profiles.append(profiles)

    coarse_labels = []
    for domain, profiles in enumerate(fine_profiles):
        count = len(profiles)
        graph = nx.Graph(); graph.add_nodes_from(range(count))
        similarity = profiles @ profiles.T
        for left in range(count):
            nearest = torch.topk(similarity[left], min(6, count)).indices.tolist()
            for right in nearest:
                if left != right and float(similarity[left, right]) > 0:
                    graph.add_edge(left, right, weight=float(similarity[left, right]))
        modules = (
            nx.community.louvain_communities(
                graph, weight="weight", resolution=0.7,
                seed=seed + 7919 * (domain + 1),
            )
            if graph.number_of_edges() else [{node} for node in graph.nodes]
        )
        labels = torch.empty(count, dtype=torch.long)
        for community, nodes in enumerate(modules):
            labels[list(nodes)] = community
        coarse_labels.append(labels)
    coarse_count = max(int(labels.max()) + 1 for labels in coarse_labels)
    shape = (corpus.num_domains, corpus.num_users + 1, feature_dim)
    user_features = torch.zeros((*shape[:-1], feature_dim * 2))
    memberships = torch.zeros(corpus.num_domains, corpus.num_users + 1, fine_count)
    hierarchy = torch.zeros(corpus.num_domains, fine_count, coarse_count)
    collaborative_features = torch.zeros(shape)
    transition_features = torch.zeros(shape)
    for domain, ((users, collaborative, transition, states, _, _), labels, parents) in enumerate(
        zip(learned, domain_labels, coarse_labels)
    ):
        index = torch.tensor(users)
        user_features[domain, index] = states
        collaborative_features[domain, index] = collaborative
        transition_features[domain, index] = transition
        memberships[domain, index] = F.one_hot(labels, fine_count).float()
        hierarchy[domain, :len(parents)] = F.one_hot(parents, coarse_count).float()
    return {
        "memberships": memberships,
        "coarse_memberships": torch.einsum("duk,dkc->duc", memberships, hierarchy),
        "fine_to_coarse": hierarchy,
        "fine_graph_profiles": torch.stack([
            F.pad(profile, (0, 0, 0, fine_count - len(profile)))
            for profile in fine_profiles
        ]),
        "fused_user_features": user_features,
        "collaborative_user_features": collaborative_features,
        "transition_user_features": transition_features,
        "fine_communities": fine_count,
        "coarse_communities": coarse_count,
        "formation": "train_only_dual_view_gnn_louvain",
        "selection_diagnostics": selection_diagnostics,
    }


def _top(counter: Mapping[str, float], limit: int) -> list[dict[str, Any]]:
    total = sum(counter.values()) or 1.0
    return [
        {"text": text, "weight": round(weight / total, 6)}
        for text, weight in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    ]


def _community_evidence(
    corpus: CanonicalCorpus,
    domain_entries: Sequence[Mapping[str, Any]],
    memberships: torch.Tensor,
    metadata: Mapping[int, Mapping[str, str]],
    threshold: float,
    *,
    level: str = "fine",
    parent_by_community: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    communities = memberships.shape[-1]
    for domain in range(corpus.num_domains):
        active_users = memberships[domain].sum(dim=1).gt(0).nonzero(as_tuple=False).flatten()
        domain_name = str(domain_entries[domain]["name"])
        for community in range(communities):
            if not memberships[domain, :, community].sum().gt(0):
                continue
            items: defaultdict[str, float] = defaultdict(float)
            categories: defaultdict[str, float] = defaultdict(float)
            transitions: defaultdict[str, float] = defaultdict(float)
            contents: defaultdict[str, float] = defaultdict(float)
            assigned, mass = 0, 0.0
            for user_id in active_users.tolist():
                affiliation = memberships[domain, user_id]
                weight = float(affiliation[community])
                # Hard assignments contain exact zeros. A zero-weight user
                # must never be counted merely because topk() returns one of
                # the tied zero entries.
                if weight <= 0.0:
                    continue
                if weight < threshold and community not in affiliation.topk(min(2, communities)).indices:
                    continue
                assigned += 1
                mass += weight
                events = [event for event in corpus.train_sequences[user_id]
                          if event.domain_id == domain]
                for event in events:
                    item = metadata[event.item_id]
                    items[item["title"]] += weight
                    categories[item["category"]] += weight
                    if item["content"]:
                        contents[f"{item['title']}: {item['content']}"] += weight
                for left, right in zip(events, events[1:]):
                    transitions[
                        f"{metadata[left.item_id]['category']} -> "
                        f"{metadata[right.item_id]['category']}"
                    ] += weight
            community_id = f"{domain_name}:{level.capitalize()}C{community}"
            parent_id = None
            if parent_by_community is not None:
                parent_value = parent_by_community[domain][community]
                parent_id = f"{domain_name}:CoarseC{int(parent_value)}"
            rows.append({
                "schema_version": "comm_cdsr.evidence.v2",
                "community_id": community_id,
                "community_level": level,
                "parent_community_id": parent_id,
                "domain": domain_name,
                "support": {"assigned_users": assigned, "effective_membership_mass": round(mass, 6)},
                "evidence": {
                    "representative_items": _top(items, 8),
                    "interest_categories": _top(categories, 8),
                    "sequential_transitions": _top(transitions, 8),
                    "content_or_review_samples": _top(contents, 5),
                },
                "privacy": {"split": "train_only", "contains_user_ids": False,
                            "contains_raw_user_sequences": False},
            })
            for section, entries in rows[-1]["evidence"].items():
                for index, entry in enumerate(entries):
                    entry["evidence_id"] = f"{section}:{index}"
    return rows


def _schema(level: str) -> dict[str, Any]:
    common = {
        "schema_version": SCHEMA_VERSION, "community_id": "domain:community",
        "domain": "domain", "confidence": 0.0, "insufficient_evidence": False,
        "memory_slots": [{
            "trigger": "observable interest or ordered behavior condition",
            "shared_need": "evidence-supported transferable need",
            "next_interest": "abstract next-interest direction, not a held-out item",
            "not_applicable": "unsupported or contradictory situation",
            "evidence_ids": ["sequential_transitions:0"],
        }],
    }
    if level == "coarse":
        return {**common, "community_level": "coarse", "transferable_core": {
            "stable_interests": ["domain-agnostic preference"],
            "underlying_needs": ["shared need"],
            "temporal_preference_patterns": ["stable sequential pattern"],
            "next_item_intent_signals": ["abstract likely next-interest direction"],
            "cross_domain_transfer_cues": ["why this knowledge may transfer"],
            "positive_membership_cues": ["supporting cue"],
            "negative_membership_cues": ["exclusion cue"],
        }, "domain_specific_residual": {
            "item_attributes": ["domain-specific attribute"],
            "consumption_contexts": ["domain-specific context"],
            "local_behavior_patterns": ["local behavior"],
            "domain_specific_next_item_signals": ["local next-interest direction"],
            "positive_membership_cues": ["supporting cue"],
            "negative_membership_cues": ["exclusion cue"],
        }, "transferable_confidence": 0.0, "domain_specific_confidence": 0.0}
    return {**common, "community_level": "fine", "parent_community_id": "domain:CoarseC0",
            "fine_interest_summary": "difference from the parent community",
            "distinctive_preferences": ["fine preference"],
            "distinctive_behavior_patterns": ["fine behavior"],
            "short_term_intent_signals": ["recent transition or intent cue"],
            "likely_next_interest_directions": ["abstract next-interest direction"],
            "positive_membership_cues": ["supporting cue"],
            "negative_membership_cues": ["exclusion cue"],
            "transferability_prior": 0.0}


def _validate(value: Any, evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("community profile must be a JSON object")
    level = str(evidence.get("community_level", "fine"))
    required = set(_schema(level))
    if required.difference(value):
        raise ValueError(f"community profile lacks {sorted(required.difference(value))}")
    if value["schema_version"] != SCHEMA_VERSION or value.get("community_level") != level:
        raise ValueError("schema_version mismatch")
    if value["community_id"] != evidence["community_id"] or value["domain"] != evidence["domain"]:
        raise ValueError("community identity mismatch")
    confidence = float(value["confidence"])
    if not 0.0 <= confidence <= 1.0 or not isinstance(value["insufficient_evidence"], bool):
        raise ValueError("confidence or insufficient_evidence is invalid")
    result = json.loads(json.dumps(value, ensure_ascii=False))
    slots = result["memory_slots"]
    allowed = {entry["evidence_id"] for entries in evidence["evidence"].values()
               for entry in entries}
    if not isinstance(slots, list) or len(slots) > 3:
        raise ValueError("memory_slots must contain at most three rules")
    for slot in slots:
        for key in ("trigger", "shared_need", "next_interest", "not_applicable"):
            if not isinstance(slot.get(key), str) or not slot[key].strip():
                raise ValueError(f"memory slot lacks {key}")
            slot[key] = _safe_text(slot[key], 240)
        refs = slot.get("evidence_ids")
        if (not isinstance(refs, list) or not refs
                or any(not isinstance(ref, str) or ref not in allowed for ref in refs)):
            raise ValueError("memory slot must cite existing evidence_ids")
    if level == "coarse":
        for section, keys in {
            "transferable_core": ("stable_interests", "underlying_needs",
                                  "temporal_preference_patterns", "next_item_intent_signals",
                                  "cross_domain_transfer_cues",
                                  "positive_membership_cues", "negative_membership_cues"),
            "domain_specific_residual": ("item_attributes", "consumption_contexts",
                                         "local_behavior_patterns",
                                         "domain_specific_next_item_signals",
                                         "positive_membership_cues", "negative_membership_cues"),
        }.items():
            if not isinstance(result.get(section), Mapping) or not all(
                isinstance(result[section].get(key), list) for key in keys
            ):
                raise ValueError(f"{section} must contain list fields")
            for key in keys:
                result[section][key] = [_safe_text(text, 140)
                                        for text in result[section][key][:8]]
        for key in ("transferable_confidence", "domain_specific_confidence"):
            result[key] = float(result[key])
            if not 0.0 <= result[key] <= 1.0:
                raise ValueError(f"{key} is invalid")
    else:
        if result.get("parent_community_id") != evidence.get("parent_community_id"):
            raise ValueError("fine community parent mismatch")
        if not isinstance(result.get("fine_interest_summary"), str):
            raise ValueError("fine_interest_summary must be text")
        for key in ("distinctive_preferences", "distinctive_behavior_patterns",
                    "short_term_intent_signals", "likely_next_interest_directions",
                    "positive_membership_cues", "negative_membership_cues"):
            if not isinstance(result.get(key), list):
                raise ValueError(f"{key} must be a list")
            result[key] = [_safe_text(text, 140) for text in result[key][:8]]
        result["transferability_prior"] = float(result["transferability_prior"])
        if not 0.0 <= result["transferability_prior"] <= 1.0:
            raise ValueError("transferability_prior is invalid")
    return result


def _prompt(evidence: Mapping[str, Any], settings: Mapping[str, Any]) -> list[dict[str, str]]:
    system = str(settings.get("system_prompt", "")).strip()
    template = str(settings.get("user_prompt_template", "")).strip()
    if not system or not template:
        raise ValueError("community_memory requires system_prompt and user_prompt_template")
    system += (
        "\nReturn at most THREE conditional memory_slots. Each has trigger, shared_need, "
        "next_interest, not_applicable and existing evidence_ids. Use abstract shared "
        "concepts, preserve transition direction, and cite evidence for every rule. "
        "Use an empty list when evidence is insufficient. Do not invent counts or "
        "probabilities. Numeric support is computed by the program."
    )
    user = template.replace(
        "{{EVIDENCE_JSON}}", json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    ).replace(
        "{{OUTPUT_SCHEMA_JSON}}", json.dumps(
            _schema(str(evidence.get("community_level", "fine"))),
            ensure_ascii=False, separators=(",", ":"),
        )
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _profile_text(value: Mapping[str, Any], kind: str) -> str:
    if value["community_level"] == "coarse":
        section = value["transferable_core" if kind == "shared"
                        else "domain_specific_residual"]
        return "; ".join(f"{key}: {', '.join(values)}" for key, values in section.items())
    pieces = [value["fine_interest_summary"],
              "preferences: " + ", ".join(value["distinctive_preferences"]),
              "behaviors: " + ", ".join(value["distinctive_behavior_patterns"]),
              "short-term intent: " + ", ".join(value["short_term_intent_signals"]),
              "next-interest directions: " + ", ".join(
                  value["likely_next_interest_directions"]
              )]
    if kind == "shared":
        pieces.append("positive cues: " + ", ".join(value["positive_membership_cues"]))
    else:
        pieces.append("negative cues: " + ", ".join(value["negative_membership_cues"]))
    return "; ".join(pieces)


class _API:
    def __init__(self, settings: Mapping[str, Any], usage_file: Path):
        env_name = str(settings.get("api_key_env", "LLM4CDSR_API_KEY"))
        key = str(settings.get("api_key", "") or "").strip() or os.environ.get(env_name, "").strip()
        if not key:
            raise ValueError(f"CommCDSR profile cache is absent; fill api_key or export {env_name}")
        base = str(settings.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        self.chat_url, self.embedding_url = base + "/chat/completions", base + "/embeddings"
        self.chat_model = str(settings.get("chat_model", "gpt-4o-mini"))
        self.embedding_model = str(settings.get("embedding_model", "text-embedding-3-small"))
        self.dimension = int(settings.get("embedding_dimension", 768))
        self.timeout = float(settings.get("timeout_seconds", 120))
        self.retries = int(settings.get("max_retries", 8))
        self.retry_max = float(settings.get("retry_max_seconds", 60))
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.usage = TokenUsageTracker(usage_file)

    def _post(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = requests.post(url, headers=self.headers, json=dict(payload), timeout=self.timeout)
                if response.ok:
                    value = response.json()
                    self.usage.record(
                        value, model=str(payload.get("model", "unknown")),
                        operation="embedding" if url == self.embedding_url else "chat",
                    )
                    return value
                message = re.sub(r"sk-[^\s\"',.]+", "sk-***", response.text[:1000])
                error = RuntimeError(f"CommCDSR API HTTP {response.status_code}: {message}")
                if 400 <= response.status_code < 500 and response.status_code not in {408, 409, 429}:
                    raise error
                last_error = error
            except RuntimeError:
                raise
            except (requests.RequestException, ValueError, KeyError) as error:
                last_error = error
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, self.retry_max) + random.uniform(0, 0.5))
        raise RuntimeError(f"CommCDSR API failed after retries: {last_error}") from last_error

    def profile(self, messages: Sequence[Mapping[str, str]], evidence: Mapping[str, Any]) -> dict[str, Any]:
        active = list(messages)
        for attempt in range(self.retries):
            response = self._post(self.chat_url, {
                "model": self.chat_model, "messages": active, "temperature": 0,
                "response_format": {"type": "json_object"},
            })
            content = str(response["choices"][0]["message"]["content"])
            try:
                return _validate(json.loads(content), evidence)
            except (ValueError, json.JSONDecodeError) as error:
                if attempt + 1 == self.retries:
                    raise RuntimeError(f"Invalid CommCDSR profile JSON: {error}") from error
                active = [*messages, {"role": "assistant", "content": content}, {
                    "role": "user", "content": f"Validation failed: {error}. Return corrected JSON only.",
                }]
        raise AssertionError("unreachable")

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        response = self._post(self.embedding_url, {
            "model": self.embedding_model, "input": list(texts),
            "dimensions": self.dimension, "encoding_format": "float",
        })
        rows = sorted(response["data"], key=lambda row: int(row["index"]))
        result = torch.tensor([row["embedding"] for row in rows], dtype=torch.float32)
        if result.shape != (len(texts), self.dimension):
            raise ValueError(f"Unexpected profile embedding shape {tuple(result.shape)}")
        return F.normalize(result, dim=1)


def _deterministic_embeddings(texts: Sequence[str], dimension: int) -> torch.Tensor:
    values = torch.zeros((len(texts), dimension), dtype=torch.float32)
    for row, text in enumerate(texts):
        for token in re.findall(r"\w+", text.lower()):
            values[row, _bucket(token, dimension)] += 1.0
    return F.normalize(values, dim=1)








def _item_community_edges(
    corpus: CanonicalCorpus, memberships: torch.Tensor, min_events: int = 0,
) -> torch.Tensor:
    """Weighted item--fine-community edges in the explicit hierarchy.

    ``min_events > 0`` removes long-tail items with fewer observed training
    events from the community edges: their membership is an average over too
    few users to be reliable, so they keep only content/id representations.
    """

    result = torch.zeros(
        (corpus.num_domains, corpus.num_items + 1, memberships.shape[-1]),
        dtype=torch.float32,
    )
    counts = torch.zeros((corpus.num_domains, corpus.num_items + 1), dtype=torch.long)
    for user_id, events in corpus.train_sequences.items():
        for event in events:
            result[event.domain_id, event.item_id] += memberships[event.domain_id, user_id]
            counts[event.domain_id, event.item_id] += 1
    if min_events > 0:
        keep = counts.ge(min_events).unsqueeze(-1)
        result = result * keep
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _expand_multi_interest_memberships(
    memberships: torch.Tensor, hierarchy_data: Mapping[str, torch.Tensor],
    *, k: int, temperature: float,
) -> torch.Tensor:
    """Promote hard Louvain labels to top-K soft multi-interest memberships.

    A user's K interests are the K communities whose fused dual-view centroid
    is most similar to the user's own state; weights are a temperature softmax
    over those cosine similarities, so they sum to one per user. The Louvain
    label always stays inside the support — no community ever loses members —
    but the mass follows the content view, which is exactly what users whose
    interests span communities need.
    """
    if k < 2:
        return memberships
    if temperature <= 0.0:
        raise ValueError("membership_temperature must be positive")
    fused = hierarchy_data.get("fused_user_features")
    if fused is None:
        fused = F.normalize(
            0.5 * hierarchy_data["collaborative_user_features"]
            + 0.5 * hierarchy_data["transition_user_features"], dim=-1,
        )
    centroids = hierarchy_data["fine_graph_profiles"]
    result = memberships.clone()
    for domain in range(memberships.shape[0]):
        active = memberships[domain].sum(dim=1).gt(0).nonzero(as_tuple=False).flatten()
        if not len(active):
            continue
        present = centroids[domain].norm(dim=-1).gt(0)
        keep = min(k, int(present.sum()))
        if keep < 2:
            continue
        similarity = F.normalize(fused[domain, active], dim=-1) @ F.normalize(
            centroids[domain], dim=-1,
        ).T
        similarity = similarity.masked_fill(~present.unsqueeze(0), -1e9)
        top = similarity.topk(keep, dim=1)
        weights = torch.softmax(top.values / temperature, dim=1)
        primary = memberships[domain, active].argmax(dim=1)
        missing = ~(top.indices.eq(primary.unsqueeze(1)).any(dim=1))
        if bool(missing.any()):
            # Keep the structural anchor: swap the weakest interest for the
            # Louvain label and rescore so weights remain a true softmax.
            rows = missing.nonzero(as_tuple=False).flatten()
            top.indices[rows, -1] = primary[rows]
            weights[rows] = torch.softmax(
                similarity[rows].gather(1, top.indices[rows]) / temperature, dim=1,
            )
        soft = torch.zeros_like(memberships[domain, active])
        soft.scatter_(1, top.indices, weights)
        result[domain, active] = soft
    return result


def prepare_comm_cdsr_features(
    corpus: CanonicalCorpus,
    domain_entries: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    *,
    communities: int | None,
    seed: int,
    logger=None,
) -> CommCDSRFeatures:
    mode = str(settings.get("mode", "api"))
    if mode not in {"api", "deterministic_test"}:
        raise ValueError("community_memory.mode must be api or deterministic_test")
    feature_dim = int(settings.get("discovery_feature_dim", 256))
    assignment = str(settings.get("community_assignment", "soft"))
    clustering_iterations = int(settings.get("clustering_iterations", 50))
    min_cluster_fraction = float(settings.get("min_cluster_fraction", 0.01))
    if assignment == "gnn_modularity":
        formation_settings = dict(
            seed=seed, feature_dim=feature_dim,
            gnn_epochs=int(settings.get("gnn_epochs", 12)),
            gnn_layers=int(settings.get("gnn_layers", 2)),
            modularity_resolution=float(settings.get("modularity_resolution", 1.0)),
            minimum_community_size=(
                int(settings["minimum_community_size"])
                if settings.get("minimum_community_size") is not None else None
            ),
        )
        formation_hash = hashlib.sha256(json.dumps(
            {"version": PROFILE_VERSION, "settings": formation_settings,
             "users": corpus.num_users, "items": corpus.num_items,
             "domains": corpus.domain_to_id}, sort_keys=True
        ).encode())
        for user, events in sorted(corpus.train_sequences.items()):
            formation_hash.update(repr((user, [(event.item_id, event.domain_id, event.timestamp)
                                               for event in events])).encode())
        formation_dir = Path(settings["cache_dir"]).resolve() / "formation"
        formation_dir.mkdir(parents=True, exist_ok=True)
        formation_file = formation_dir / (formation_hash.hexdigest()[:20] + ".pt")
        if formation_file.exists():
            hierarchy_data = torch.load(formation_file, map_location="cpu", weights_only=True)
        else:
            hierarchy_data = build_gnn_modularity_hierarchy(corpus, **formation_settings)
            temporary = formation_file.with_suffix(f".{os.getpid()}.tmp")
            torch.save(hierarchy_data, temporary)
            temporary.replace(formation_file)
        communities = int(hierarchy_data["fine_communities"])
        coarse_count = int(hierarchy_data["coarse_communities"])
    else:
        if communities is None:
            raise ValueError("fixed community discovery requires an explicit community count")
        coarse_count = int(settings.get("coarse_communities", min(8, communities)))
        if communities > 2 and coarse_count >= communities:
            coarse_count = max(2, communities // 2)
        coarse_count = max(2, min(coarse_count, communities))
        hierarchy_data = build_comm_cdsr_hierarchy(
            corpus, communities, coarse_count, seed=seed, feature_dim=feature_dim,
            temperature=float(settings.get("discovery_temperature", 0.2)),
            assignment=assignment, clustering_iterations=clustering_iterations,
            min_cluster_fraction=min_cluster_fraction,
        )
    multi_interest_k = int(settings.get("multi_interest_k", 1))
    multi_interest = multi_interest_k > 1
    if multi_interest:
        hierarchy_data["memberships"] = _expand_multi_interest_memberships(
            hierarchy_data["memberships"], hierarchy_data,
            k=multi_interest_k,
            temperature=float(settings.get("membership_temperature", 0.15)),
        )
        hierarchy_data["coarse_memberships"] = torch.einsum(
            "duk,dkc->duc", hierarchy_data["memberships"],
            hierarchy_data["fine_to_coarse"],
        )
    metadata = _item_metadata(corpus, domain_entries)
    threshold = float(settings.get("overlap_membership_threshold", 0.2))
    coarse_evidence = _community_evidence(
        corpus, domain_entries, hierarchy_data["coarse_memberships"], metadata,
        threshold, level="coarse",
    )
    parents = hierarchy_data["fine_to_coarse"].argmax(dim=-1)
    fine_evidence = _community_evidence(
        corpus, domain_entries, hierarchy_data["memberships"], metadata,
        threshold, level="fine", parent_by_community=parents,
    )
    evidence = coarse_evidence + fine_evidence
    identity = {
        "profile_version": PROFILE_VERSION, "schema_version": SCHEMA_VERSION, "mode": mode,
        "chat_model": settings.get("chat_model"), "embedding_model": settings.get("embedding_model"),
        "embedding_dimension": int(settings.get("embedding_dimension", 768)),
        "coarse_communities": coarse_count, "fine_communities": communities,
        "community_assignment": assignment,
        "clustering_iterations": clustering_iterations,
        "min_cluster_fraction": min_cluster_fraction,
        "gnn_epochs": int(settings.get("gnn_epochs", 12)),
        "gnn_layers": int(settings.get("gnn_layers", 2)),
        "modularity_resolution": float(settings.get("modularity_resolution", 1.0)),
        "minimum_community_size": settings.get("minimum_community_size"),
        "hierarchy_digest": hashlib.sha256(b"".join(
            tensor.detach().cpu().contiguous().numpy().tobytes()
            for tensor in (
                hierarchy_data["memberships"], hierarchy_data["fine_to_coarse"],
                hierarchy_data["fine_graph_profiles"],
            )
        )).hexdigest(),
        "system_prompt": settings.get("system_prompt"),
        "user_prompt_template": settings.get("user_prompt_template"), "evidence": evidence,
    }
    if multi_interest:
        identity["multi_interest_k"] = multi_interest_k
        identity["membership_temperature"] = float(
            settings.get("membership_temperature", 0.15)
        )
    # Evidence-quality filters (long-tail / weak-slot cleaning). They are only
    # hashed into the fingerprint when active, so default runs keep the exact
    # legacy fingerprint and reuse their caches untouched.
    filters = {}
    slot_floor = float(settings.get("slot_support_floor", 0.0))
    if slot_floor > 0.0:
        filters["slot_support_floor"] = slot_floor
    item_min_events = int(settings.get("item_min_events", 0))
    if item_min_events > 0:
        filters["item_min_events"] = item_min_events
    base_fingerprint = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    if filters:
        identity["community_evidence_filters"] = filters
    fingerprint = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    destination = Path(settings["cache_dir"]).resolve() / fingerprint
    destination.mkdir(parents=True, exist_ok=True)
    if filters:
        # Reuse the unfiltered run's LLM products verbatim: filtered variants
        # must differ from the reference ONLY through the filters themselves,
        # never through freshly regenerated profile text.
        base_destination = Path(settings["cache_dir"]).resolve() / base_fingerprint
        for name in ("profiles.jsonl", "embeddings.jsonl"):
            source, target = base_destination / name, destination / name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
        if logger:
            logger.info(
                "CommCDSR evidence filters active: %s (profiles reused from %s)",
                filters, base_fingerprint,
            )
    feature_file, manifest_file = destination / "features.pt", destination / "manifest.json"
    usage_file = destination / "api_usage.json"
    if feature_file.is_file() and manifest_file.is_file():
        saved = torch.load(feature_file, map_location="cpu", weights_only=True)
        return CommCDSRFeatures(
            memberships=saved["memberships"],
            coarse_memberships=saved["coarse_memberships"],
            fine_to_coarse=saved["fine_to_coarse"],
            item_fine_memberships=saved["item_fine_memberships"],
            fine_graph_profiles=saved["fine_graph_profiles"],
            fine_shared_profiles=saved["fine_shared_profiles"],
            fine_private_profiles=saved["fine_private_profiles"],
            memory_keys=saved["memory_keys"], memory_values=saved["memory_values"],
            memory_support=saved["memory_support"],
            concept_behavior=saved["concept_behavior"],
            structure_profiles=saved["structure_profiles"],
            source="profile_cache", cache_dir=destination,
            manifest=json.loads(manifest_file.read_text(encoding="utf-8")),
            api_usage={}, api_usage_cumulative=TokenUsageTracker(usage_file).snapshot(),
        )
    (destination / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    coarse_responses: list[dict[str, Any]] = []
    fine_responses: list[dict[str, Any]] = []
    api = None
    if mode == "api":
        api = _API(settings, usage_file)
        response_file = destination / "profiles.jsonl"
        cached: dict[str, dict[str, Any]] = {}
        if response_file.exists():
            for line in response_file.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                    cached[str(value["community_id"])] = value
                except (ValueError, KeyError):
                    continue
        responses_by_id: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(evidence, start=1):
            value = cached.get(str(row["community_id"]))
            try:
                value = _validate(value, row) if value is not None else None
            except ValueError:
                value = None
            if value is None:
                prompt_evidence = dict(row)
                if row["community_level"] == "fine":
                    parent = responses_by_id.get(str(row["parent_community_id"]))
                    if parent is None:
                        raise RuntimeError("Fine community was prepared before its coarse parent")
                    prompt_evidence["parent_profile"] = parent
                value = api.profile(_prompt(prompt_evidence, settings), row)
                with response_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            responses_by_id[str(row["community_id"])] = value
            (coarse_responses if row["community_level"] == "coarse"
             else fine_responses).append(value)
            if logger:
                logger.info(
                    "CommCDSR %s community profile %d/%d complete",
                    row["community_level"], index, len(evidence),
                )
    else:
        for row in evidence:
            categories = [value["text"] for value in row["evidence"]["interest_categories"][:4]]
            transitions = [value["text"] for value in row["evidence"]["sequential_transitions"][:4]]
            common = {"schema_version": SCHEMA_VERSION, "community_id": row["community_id"],
                      "community_level": row["community_level"], "domain": row["domain"],
                      "confidence": 0.5, "insufficient_evidence": not bool(categories)}
            common["memory_slots"] = [{
                "trigger": entry["text"].split(" -> ")[0],
                "shared_need": entry["text"].split(" -> ")[0],
                "next_interest": entry["text"].split(" -> ")[-1],
                "not_applicable": "insufficient matching history",
                "evidence_ids": [entry["evidence_id"]],
            } for entry in row["evidence"]["sequential_transitions"][:3]]
            if row["community_level"] == "coarse":
                value = {**common, "transferable_core": {
                    "stable_interests": categories, "underlying_needs": categories[:2],
                    "temporal_preference_patterns": transitions,
                    "next_item_intent_signals": transitions[-2:],
                    "cross_domain_transfer_cues": categories[:2],
                    "positive_membership_cues": categories, "negative_membership_cues": [],
                }, "domain_specific_residual": {
                    "item_attributes": categories, "consumption_contexts": [],
                    "local_behavior_patterns": transitions,
                    "domain_specific_next_item_signals": transitions[-2:],
                    "positive_membership_cues": categories, "negative_membership_cues": [],
                }, "transferable_confidence": 0.5, "domain_specific_confidence": 0.5}
                coarse_responses.append(value)
            else:
                value = {**common, "parent_community_id": row["parent_community_id"],
                         "fine_interest_summary": ", ".join(categories) or "general interest",
                         "distinctive_preferences": categories,
                         "distinctive_behavior_patterns": transitions,
                         "short_term_intent_signals": transitions[-2:],
                         "likely_next_interest_directions": categories[:2],
                         "positive_membership_cues": categories,
                         "negative_membership_cues": [], "transferability_prior": 0.5}
                fine_responses.append(value)
    dimension = int(settings.get("embedding_dimension", 768))
    # Only transferable/shared text enters the numerical bridge memory.
    # Domain-private text remains in the cached JSON for auditability and
    # membership cues, but embedding it would waste tokens and risk accidental
    # cross-domain transport.
    texts = ([_profile_text(value, "shared") for value in coarse_responses]
             + [_profile_text(value, "shared") for value in fine_responses])
    slot_records = []
    for value in fine_responses:
        for slot in value["memory_slots"]:
            slot_records.append((value["community_id"], slot, len(texts)))
            texts.extend([slot["trigger"], slot["shared_need"] + "; " + slot["next_interest"]])
    # Category endpoints share the SAME text encoder across domains. Unlike
    # local graph coordinates, these directional moments are comparable.
    categories = sorted({metadata[event.item_id]["category"]
                         for events in corpus.train_sequences.values() for event in events})
    category_offset = len(texts)
    texts.extend(categories)
    if api is None:
        embedded = _deterministic_embeddings(texts, dimension)
        (destination / "profiles.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in coarse_responses + fine_responses), encoding="utf-8"
        )
    else:
        batch_size = max(1, int(settings.get("embedding_batch_size", 32)))
        # Append each completed batch so a later failure never repeats it.
        embedding_file = destination / "embeddings.jsonl"
        cached_embeddings = {}
        if embedding_file.exists():
            for line in embedding_file.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if len(row["vector"]) == dimension:
                        cached_embeddings[row["text"]] = row["vector"]
                except (ValueError, KeyError, TypeError):
                    continue
        pending = list(dict.fromkeys(text for text in texts if text not in cached_embeddings))
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            vectors = api.embed(batch)
            with embedding_file.open("a", encoding="utf-8") as stream:
                for text, vector in zip(batch, vectors.tolist()):
                    cached_embeddings[text] = vector
                    stream.write(json.dumps({"text": text, "vector": vector}, ensure_ascii=False) + "\n")
        embedded = torch.tensor([cached_embeddings[text] for text in texts], dtype=torch.float32)
    coarse_shared = torch.zeros(corpus.num_domains, coarse_count, dimension)
    fine_shared_delta = torch.zeros(corpus.num_domains, communities, dimension)
    transferability = torch.zeros(corpus.num_domains, communities, 1)
    domain_ids = {str(entry["name"]): index for index, entry in enumerate(domain_entries)}
    positions = {}
    for offset, value in enumerate(coarse_responses + fine_responses):
        domain = domain_ids[value["domain"]]
        community = int(value["community_id"].rsplit("C", 1)[1])
        positions[value["community_id"]] = (domain, community)
        if offset < len(coarse_responses):
            coarse_shared[domain, community] = embedded[offset]
        else:
            fine_shared_delta[domain, community] = embedded[offset]
            transferability[domain, community] = float(value["transferability_prior"])
    inherited_shared = torch.einsum(
        "dkc,dch->dkh", hierarchy_data["fine_to_coarse"], coarse_shared
    )
    fine_shared = F.normalize(
        inherited_shared + transferability * fine_shared_delta, dim=-1
    )
    fine_private = torch.zeros_like(fine_shared)
    memberships = hierarchy_data["memberships"]
    memory_keys = torch.zeros(corpus.num_domains, communities, 3, dimension)
    memory_values = torch.zeros_like(memory_keys)
    memory_support = torch.zeros(corpus.num_domains, communities, 3)
    evidence_by_id = {row["community_id"]: row for row in fine_evidence}
    slot_counts = Counter()
    dropped_slots = 0
    for community_id, slot, offset in slot_records:
        domain, community = positions[community_id]
        row = evidence_by_id[community_id]
        weights = {entry["evidence_id"]: float(entry["weight"])
                   for entries in row["evidence"].values() for entry in entries}
        # Program-derived support, never an LLM-invented probability.
        support = min(1.0, sum(weights[ref] for ref in set(slot["evidence_ids"])))
        support *= min(1.0, row["support"]["assigned_users"] / 8.0)
        # Weak-evidence slots carry more noise than signal on large-catalog
        # domains: drop them instead of passing them through with low weight.
        if support < slot_floor:
            dropped_slots += 1
            continue
        index = slot_counts[community_id]; slot_counts[community_id] += 1
        memory_keys[domain, community, index] = embedded[offset]
        memory_values[domain, community, index] = embedded[offset + 1]
        memory_support[domain, community, index] = support
    if slot_floor > 0 and logger:
        logger.info(
            "CommCDSR slot filter: kept %d/%d memory slots (support floor %.3f)",
            len(slot_records) - dropped_slots, len(slot_records), slot_floor,
        )
    category_vectors = dict(zip(categories, embedded[category_offset:]))
    concept_behavior = torch.zeros(corpus.num_domains, communities, dimension * 2 + 64)
    # A common fixed sketch retains JOINT predecessor/successor information,
    # including direction, without a vocabulary-squared transition matrix.
    projection = torch.randn(dimension, 8, generator=torch.Generator().manual_seed(0)) / math.sqrt(dimension)
    structure_profiles = torch.zeros(corpus.num_domains, communities, 4)
    transition_mass = torch.zeros(corpus.num_domains, communities)
    for user, events in corpus.train_sequences.items():
        for domain in range(corpus.num_domains):
            local = [event for event in events if event.domain_id == domain]
            if not local:
                continue
            affiliation = memberships[domain, user]
            counts = Counter(event.item_id for event in local)
            lengths = float(len(local))
            # Common interpretable statistics, no cross-domain latent cosine.
            descriptor = torch.tensor([math.log1p(lengths), len(counts) / lengths,
                                       max(counts.values()) / lengths,
                                       math.log1p(len({metadata[e.item_id]['category'] for e in local}))])
            structure_profiles[domain] += affiliation[:, None] * descriptor
            for left, right in zip(local, local[1:]):
                before = category_vectors[metadata[left.item_id]["category"]]
                after = category_vectors[metadata[right.item_id]["category"]]
                joint = torch.outer(before @ projection, after @ projection).flatten()
                directed = torch.cat((before, after, joint))
                concept_behavior[domain] += affiliation[:, None] * directed
                transition_mass[domain] += affiliation
    concept_behavior /= transition_mass.unsqueeze(-1).clamp_min(1)
    structure_profiles /= memberships.sum(dim=1).unsqueeze(-1).clamp_min(1)
    coarse_memberships = torch.einsum(
        "duk,dkc->duc", memberships, hierarchy_data["fine_to_coarse"]
    )
    fused_users = hierarchy_data.get("fused_user_features")
    if fused_users is None:
        fused_users = F.normalize(
            0.5 * hierarchy_data["collaborative_user_features"]
            + 0.5 * hierarchy_data["transition_user_features"], dim=-1,
        )
    fine_graph = torch.einsum("duk,duh->dkh", memberships, fused_users)
    fine_graph = F.normalize(
        fine_graph / memberships.sum(dim=1).unsqueeze(-1).clamp_min(1e-8), dim=-1
    )
    item_fine_memberships = _item_community_edges(
        corpus, memberships, min_events=item_min_events
    )
    if item_min_events > 0 and logger:
        per_domain = item_fine_memberships.abs().sum(dim=-1).gt(0)
        per_domain[:, 0] = False  # padding column is empty by construction
        coverage = (per_domain.sum(dim=1) / per_domain.shape[1]).tolist()
        logger.info(
            "CommCDSR item filter: min_events=%d keeps item-edge coverage %s",
            item_min_events, [round(value, 4) for value in coverage],
        )
    torch.save({
        "memberships": memberships,
        "coarse_memberships": coarse_memberships,
        "fine_to_coarse": hierarchy_data["fine_to_coarse"],
        "item_fine_memberships": item_fine_memberships,
        "fine_graph_profiles": fine_graph,
        "fine_shared_profiles": fine_shared,
        "fine_private_profiles": fine_private,
        "memory_keys": memory_keys, "memory_values": memory_values,
        "memory_support": memory_support, "concept_behavior": concept_behavior,
        "structure_profiles": structure_profiles,
    }, feature_file)
    community_sizes = []
    for domain in range(corpus.num_domains):
        active = memberships[domain].sum(dim=1).gt(0)
        community_sizes.append(
            memberships[domain, active].argmax(dim=1).bincount(
                minlength=communities
            ).tolist()
        )
    if assignment in {"hard_spherical_kmeans", "gnn_modularity"} and not multi_interest:
        active_values = memberships[memberships.sum(dim=-1).gt(0)]
        if not torch.all((active_values == 0) | (active_values == 1)):
            raise RuntimeError("hard community discovery produced non-binary memberships")
    graph_separation = []
    for domain in range(corpus.num_domains):
        cosine = fine_graph[domain] @ fine_graph[domain].T
        present = memberships[domain].sum(dim=0).gt(0)
        mask = ~torch.eye(communities, dtype=torch.bool) & present[:, None] & present[None, :]
        graph_separation.append({
            "mean_off_diagonal_cosine": float(cosine[mask].mean()) if mask.any() else 0.0,
            "max_off_diagonal_cosine": float(cosine[mask].max()) if mask.any() else 0.0,
        })
    manifest = {
        "fingerprint": fingerprint, "profile_version": PROFILE_VERSION,
        "schema_version": SCHEMA_VERSION, "source_split": "train_only",
        "profile_signal": (
            "item title/category/brand/description plus reviews from canonical "
            "train (user,item) pairs only"
        ),
        "review_leakage_guard": "canonical train_sequences (user,item) allow-list",
        "fine_communities": communities, "coarse_communities": coarse_count,
        "domains": corpus.num_domains,
        "embedding_dimension": dimension, "api_key_persisted": False,
        "formation": hierarchy_data.get("formation", f"dual_view_{assignment}_hierarchy")
        + "_fixed_membership",
        "community_sizes_by_domain": community_sizes,
        "graph_profile_separation": graph_separation,
        "module2_bridge": "shared_semantics_directed_concept_moments_structural_statistics_partial_ot",
        "memory_slots": "at_most_three_evidence_cited_conditional_rules_per_fine_community",
        "active_communities_by_domain": [sum(size > 0 for size in sizes) for sizes in community_sizes],
        "community_selection": hierarchy_data.get("selection_diagnostics", []),
        "llm_calls": {"coarse": len(coarse_responses), "fine": len(fine_responses)},
    }
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return CommCDSRFeatures(
        memberships=memberships, coarse_memberships=coarse_memberships,
        fine_to_coarse=hierarchy_data["fine_to_coarse"],
        item_fine_memberships=item_fine_memberships,
        fine_graph_profiles=fine_graph,
        fine_shared_profiles=fine_shared, fine_private_profiles=fine_private,
        memory_keys=memory_keys, memory_values=memory_values,
        memory_support=memory_support, concept_behavior=concept_behavior,
        structure_profiles=structure_profiles,
        source="api" if api is not None else "deterministic_test",
        cache_dir=destination, manifest=manifest,
        api_usage=api.usage.session_snapshot() if api else {},
        api_usage_cumulative=api.usage.snapshot() if api else {},
    )
