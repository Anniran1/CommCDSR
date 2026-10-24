"""CommCDSR: identity-free community transfer over a strong target backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from cdsr.data.schema import CanonicalCorpus
from cdsr.features.comm_cdsr_best import CommCDSRFeatures
from cdsr.models.base import CDSRModel, LossOutput


@dataclass(frozen=True)
class CommCDSRConfig:
    hidden_size: int = 128
    sequence_length: int = 100
    num_blocks: int = 2
    num_heads: int = 4
    num_negatives: int = 128
    num_communities: int = 16
    graph_feature_dim: int = 256
    dropout: float = 0.2
    community_temperature: float = 0.5
    transfer_scale: float = 1.0
    ot_epsilon: float = 0.1
    ot_iterations: int = 50
    community_mass_power: float = 0.5
    scheme_dropout: float = 0.1
    target_only_loss_weight: float = 0.5
    user_confidence_tau: float = 5.0
    bridge_topk: int = 4
    intent_prior_weight: float = 0.35
    unmatched_cost: float = 0.6
    safe_transfer_weight: float = 0.1
    safe_transfer_tolerance: float = 0.02
    backbone_warmup_epochs: int = 5
    interest_diversity_weight: float = 0.0
    community_next_item_weight: float = 0.0
    transfer_next_item_weight: float = 0.0
    community_score_weight: float = 0.0
    community_score_ramp_epochs: int = 5
    use_item_content: bool = False
    content_dropout: float = 0.0

    def __post_init__(self) -> None:
        if (self.hidden_size < 1 or self.sequence_length < 1
                or self.num_communities < 1 or self.num_blocks < 1
                or self.num_heads < 1 or self.num_negatives < 1
                or self.graph_feature_dim < 1 or self.bridge_topk < 1):
            raise ValueError("CommCDSR dimensions and sampling settings are invalid")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.ot_epsilon <= 0 or self.ot_iterations < 1:
            raise ValueError("CommCDSR OT settings are invalid")
        if self.user_confidence_tau <= 0:
            raise ValueError("user_confidence_tau must be positive")
        if min(self.unmatched_cost, self.safe_transfer_weight, self.safe_transfer_tolerance,
               self.interest_diversity_weight, self.community_next_item_weight,
               self.transfer_next_item_weight, self.community_score_weight,
               self.content_dropout) < 0:
            raise ValueError("transfer cost, weight and tolerance must be nonnegative")
        for name in ("transfer_scale", "community_mass_power", "scheme_dropout",
                     "intent_prior_weight"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


def _sinkhorn(cost: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
              epsilon: float, iterations: int) -> torch.Tensor:
    kernel = torch.exp(-cost / epsilon).clamp_min(1e-30)
    u, v = torch.ones_like(left), torch.ones_like(right)
    for _ in range(iterations):
        u = left / (kernel @ v).clamp_min(1e-12)
        v = right / (kernel.T @ u).clamp_min(1e-12)
    return u.unsqueeze(1) * kernel * v.unsqueeze(0)


def _community_marginal(membership: torch.Tensor, power: float) -> torch.Tensor:
    counts = membership[1:].sum(dim=0)
    mass = counts.clamp_min(1e-12).pow(power) * counts.gt(0)
    return mass / mass.sum().clamp_min(1e-12)


def _ot_bridge(
    target: torch.Tensor,
    source: torch.Tensor,
    target_membership: torch.Tensor,
    source_membership: torch.Tensor,
    target_hierarchy: torch.Tensor,
    source_hierarchy: torch.Tensor,
    config: CommCDSRConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectangular OT prior with a per-row null alternative; padding has zero mass.

    Source capacity is a prior, not a requirement to transfer. A null competitor
    compares absolute cost to unmatched_cost; rejected mass is NOT renormalized.
    """

    target, source = F.normalize(target.float(), dim=-1), F.normalize(source.float(), dim=-1)
    left = _community_marginal(target_membership, config.community_mass_power)
    right = _community_marginal(source_membership, config.community_mass_power)
    cost = (1.0 - target @ source.T).clamp_min(0.0)
    plan = _sinkhorn(cost, left, right, config.ot_epsilon, config.ot_iterations)
    bridge = plan / left.unsqueeze(1).clamp_min(1e-12)
    bridge = bridge / bridge.sum(dim=1, keepdim=True).clamp_min(1e-12)
    # A bridge should expose a small set of plausible source communities, not
    # diffuse weak mass over the complete source catalog.
    keep = min(config.bridge_topk, bridge.shape[1])
    indices = bridge.topk(keep, dim=1).indices
    sparse = torch.zeros_like(bridge).scatter_(1, indices, 1.0)
    bridge = bridge * sparse
    bridge = bridge / bridge.sum(dim=1, keepdim=True).clamp_min(1e-12)
    available = source.norm(dim=-1).gt(0) & right.gt(0)
    best_cost = cost.masked_fill(~available.unsqueeze(0), 1e4).min(dim=1).values
    accepted = torch.sigmoid((config.unmatched_cost - best_cost) / config.ot_epsilon)
    accepted *= target.norm(dim=-1).gt(0) & left.gt(0)
    entropy = -(bridge.clamp_min(1e-12) * bridge.clamp_min(1e-12).log()).sum(dim=1)
    certainty = 1.0 - entropy / float(torch.log(torch.tensor(max(int(available.sum()), 2))))
    reliability = accepted * certainty.clamp(0, 1)
    return bridge * accepted[:, None], reliability.clamp(0.0, 1.0)




class _PointWiseFeedForward(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden, hidden, 1)
        self.conv2 = nn.Conv1d(hidden, hidden, 1)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.dropout2(self.conv2(F.relu(
            self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        ))).transpose(-1, -2)
        return values + inputs


class _CausalAttentionEncoder(nn.Module):
    """Causal ATTENTION block matching SemaCDR's sequence encoder."""

    def __init__(self, hidden: int, length: int, blocks: int, heads: int, dropout: float):
        super().__init__()
        self.length = length
        self.position = nn.Embedding(length, hidden, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.attention_norms = nn.ModuleList(nn.LayerNorm(hidden, eps=1e-8) for _ in range(blocks))
        self.attentions = nn.ModuleList(nn.MultiheadAttention(hidden, heads, dropout) for _ in range(blocks))
        self.forward_norms = nn.ModuleList(nn.LayerNorm(hidden, eps=1e-8) for _ in range(blocks))
        self.forwards = nn.ModuleList(_PointWiseFeedForward(hidden, dropout) for _ in range(blocks))
        self.last_norm = nn.LayerNorm(hidden, eps=1e-8)

    def forward(self, embeddings: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)
        sequence = self.embedding_dropout(embeddings + self.position(positions).unsqueeze(0))
        sequence = sequence * ~padding.unsqueeze(-1)
        causal = ~torch.tril(torch.ones(
            embeddings.shape[1], embeddings.shape[1], dtype=torch.bool,
            device=embeddings.device,
        ))
        for attn_norm, attention, ff_norm, feed_forward in zip(
            self.attention_norms, self.attentions, self.forward_norms, self.forwards,
        ):
            transposed = sequence.transpose(0, 1)
            query = attn_norm(transposed)
            attended, _ = attention(query, transposed, transposed,
                                    attn_mask=causal, need_weights=False)
            sequence = feed_forward(ff_norm((query + attended).transpose(0, 1)))
            sequence = sequence * ~padding.unsqueeze(-1)
        return self.last_norm(sequence) * ~padding.unsqueeze(-1)


class CommCDSRModel(CDSRModel):
    """Target recommender whose only cross-domain path is community memory."""

    def __init__(self, corpus: CanonicalCorpus, target_domain_id: int,
                 features: CommCDSRFeatures, config: CommCDSRConfig | None = None,
                 item_content: torch.Tensor | None = None):
        super().__init__()
        self.adapter_config = config or CommCDSRConfig()
        self.target_domain_id = int(target_domain_id)
        self.num_domains = corpus.num_domains
        self.sequence_length = self.adapter_config.sequence_length
        hidden, communities = self.adapter_config.hidden_size, self.adapter_config.num_communities
        expected = (self.num_domains, corpus.num_users + 1, communities)
        if tuple(features.memberships.shape) != expected:
            raise ValueError(f"community memberships {tuple(features.memberships.shape)} != {expected}")
        if features.fine_graph_profiles.shape[-1] != self.adapter_config.graph_feature_dim:
            raise ValueError("community graph feature dimension does not match configuration")

        profile_dim = features.fine_shared_profiles.shape[-1]
        self.item_embedding = nn.Embedding(corpus.num_items + 1, hidden, padding_idx=0)
        # The ID embedding is target-local personalization only. Its influence
        # grows with the amount of target history visible at the prediction
        # position, so cold users never receive an arbitrary random vector.
        self.target_user_embedding = nn.Embedding(
            corpus.num_users + 1, hidden, padding_idx=0
        )
        self.graph_projection = nn.Linear(self.adapter_config.graph_feature_dim, hidden, bias=False)
        self.shared_projection = nn.Linear(profile_dim, hidden, bias=False)
        self.behavior_projection = nn.Linear(profile_dim * 2 + 64, hidden, bias=False)
        self.target_encoder = _CausalAttentionEncoder(
            hidden, self.sequence_length + 1, self.adapter_config.num_blocks,
            self.adapter_config.num_heads, self.adapter_config.dropout,
        )
        self.graph_item_scale = nn.Parameter(torch.zeros(()))
        self.community_scale = nn.Parameter(torch.zeros(()))
        self.local_weight_net = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(self.adapter_config.dropout),
            nn.Linear(hidden, 3),
        )
        self.scheme_router = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden), nn.GELU(), nn.Dropout(self.adapter_config.dropout),
            nn.Linear(hidden, 3),
        )
        self.memory_key_projection = nn.Linear(profile_dim, hidden, bias=False)
        self.memory_value_projection = nn.Linear(profile_dim, hidden, bias=False)
        self.affiliation_gate = nn.Linear(hidden + 1, 1)
        nn.init.zeros_(self.affiliation_gate.weight)
        prior = min(max(self.adapter_config.intent_prior_weight, 1e-4), 1 - 1e-4)
        nn.init.constant_(self.affiliation_gate.bias, float(torch.logit(torch.tensor(prior))))
        self.bridge_gate = nn.Sequential(
            nn.Linear(hidden * 3 + 5, hidden), nn.GELU(), nn.Dropout(self.adapter_config.dropout),
            nn.Linear(hidden, 1),
        )
        self.target_norm = nn.LayerNorm(hidden)
        # Exact target-only starting point, without a different normalization path.
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.item_bias = nn.Embedding(corpus.num_items + 1, 1, padding_idx=0)

        self.register_buffer("memberships", features.memberships, persistent=True)
        self.register_buffer("fine_to_coarse", features.fine_to_coarse, persistent=True)
        self.register_buffer("item_fine_memberships", features.item_fine_memberships, persistent=True)
        self.register_buffer("raw_graph_profiles", features.fine_graph_profiles, persistent=True)
        self.register_buffer("raw_shared_profiles", features.fine_shared_profiles, persistent=True)
        behavior = (features.concept_behavior if features.concept_behavior is not None
                    else torch.zeros(self.num_domains, communities, profile_dim * 2 + 64))
        self.register_buffer("raw_behavior_profiles", behavior, persistent=True)
        self.register_buffer("community_mask", features.memberships.sum(dim=1).gt(0))
        self.register_buffer("memory_keys", features.memory_keys if features.memory_keys is not None
                             else torch.zeros(self.num_domains, communities, 3, profile_dim))
        self.register_buffer("memory_values", features.memory_values if features.memory_values is not None
                             else torch.zeros(self.num_domains, communities, 3, profile_dim))
        self.register_buffer("memory_support", features.memory_support if features.memory_support is not None
                             else torch.zeros(self.num_domains, communities, 3))
        structure = (features.structure_profiles if features.structure_profiles is not None
                     else torch.zeros(self.num_domains, communities, 4))

        bridges = torch.zeros(self.num_domains, self.num_domains, 3,
                              communities, communities)
        reliability = torch.ones(self.num_domains, self.num_domains, 3, communities)
        identity = torch.eye(communities)
        for target in range(self.num_domains):
            for source in range(self.num_domains):
                if target == source:
                    bridges[target, source] = (identity * self.community_mask[target]).unsqueeze(0)
                    continue
                # The bridges express complementary evidence: LLM-derived
                # domain-agnostic semantics, sequential behavior, and learned
                # graph structure. Domain-private LLM profiles are never
                # transported.
                for scheme, channel in enumerate((features.fine_shared_profiles,
                                                   behavior,
                                                   structure)):
                    bridge, confidence = _ot_bridge(
                        channel[target], channel[source],
                        features.memberships[target], features.memberships[source],
                        features.fine_to_coarse[target], features.fine_to_coarse[source],
                        self.adapter_config,
                    )
                    bridges[target, source, scheme] = bridge
                    reliability[target, source, scheme] = confidence
        self.register_buffer("base_bridges", bridges, persistent=True)
        self.register_buffer("bridge_reliability", reliability, persistent=True)
        self.register_buffer("community_bridges", bridges.mean(dim=2), persistent=False)
        if self.adapter_config.use_item_content:
            if item_content is None:
                raise ValueError("use_item_content requires item content embeddings")
            if int(item_content.shape[0]) != corpus.num_items + 1:
                raise ValueError("item content matrix is not aligned to the item space")
            # Per-user/per-item semantics as a zero-initialized residual: the
            # model starts from the community pipeline above and learns to use
            # textual content only as far as it helps.
            self.register_buffer("item_content", item_content, persistent=True)
            self.content_projection = nn.Linear(item_content.shape[-1], hidden, bias=False)
            self.content_dropout_layer = nn.Dropout(self.adapter_config.content_dropout)
            self.content_user_scale = nn.Parameter(torch.zeros(()))
            self.content_item_scale = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.target_user_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.target_user_embedding.weight[0].zero_()
        self.memory_source = f"{features.source}_community_profiles"
        # Evaluation-only switches used by the quality audit. They never
        # participate in training or checkpoint state.
        self.audit_bridge_scheme: int | None = None
        self.audit_disable_transfer = False
        self.audit_disable_profiles = False
        self.audit_disable_semantic_route = False
        self.register_buffer("training_epoch", torch.tensor(0))

    def set_training_epoch(self, epoch: int) -> None:
        self.training_epoch.fill_(epoch)

    def validate_batch(self, batch: Mapping[str, torch.Tensor]) -> None:
        if "comm_context" in batch:
            required = {"comm_user_id", "comm_context", "comm_positive",
                        "comm_negative", "comm_negative_mask"}
            missing = required.difference(batch)
            if missing:
                raise KeyError(f"Missing CommCDSR training fields: {sorted(missing)}")
        else:
            super().validate_batch(batch)

    def _profiles(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared_projection(self.raw_shared_profiles)
        behavior = self.behavior_projection(self.raw_behavior_profiles)
        graph = self.graph_projection(self.raw_graph_profiles)
        return shared, behavior, graph

    def _interest_diversity(self) -> torch.Tensor:
        """Mean squared off-diagonal cosine between projected community profiles.

        Multi-interest memberships only carry information while the communities
        a user mixes stay distinguishable after projection; without this term
        the learned projections can collapse neighbouring communities onto a
        shared direction and the mixture degenerates to a single interest.
        """
        total = self.raw_graph_profiles.new_zeros(())
        panels = 0
        for channel in self._profiles():
            for domain in range(self.num_domains):
                active = self.community_mask[domain]
                count = int(active.sum())
                if count < 2:
                    continue
                embedding = F.normalize(channel[domain][active], dim=-1)
                gram = embedding @ embedding.T
                off = gram - torch.eye(count, device=gram.device, dtype=gram.dtype)
                total = total + off.pow(2).sum() / float(count * (count - 1))
                panels += 1
        return total / max(panels, 1)

    def _community_next_item_loss(
        self, representation: torch.Tensor, positive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Soft cross-entropy from a representation to the next item's community.

        The target is the positive item's fine-community membership
        distribution, so the representation must place the next item inside
        the user's community landscape — the explicit community/next-item
        coupling. Returns (loss, fraction of supervised rows).
        """
        graph = self.graph_projection(self.raw_graph_profiles)
        embeddings = F.normalize(graph[self.target_domain_id], dim=-1)
        target = self.item_fine_memberships[self.target_domain_id, positive]
        supervised = target.sum(dim=-1).gt(0)
        total = representation.new_zeros(())
        if not bool(supervised.any()):
            return total, supervised.float().mean()
        distribution = target[supervised] / target[supervised].sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        logits = representation[supervised] @ embeddings.T
        logits = logits.masked_fill(
            ~self.community_mask[self.target_domain_id], -1e9
        )
        loss = -(distribution * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        return loss, supervised.float().mean()

    def _community_score(self, representation: torch.Tensor,
                         item_ids: torch.Tensor) -> torch.Tensor:
        """Community-compatibility score that enters the ranking margin.

        The fused state predicts a next-community distribution; a candidate
        item contributes the inner product with its frozen fine-community
        membership. Unlike the auxiliary coupling loss, this term shifts the
        scores of positives and negatives directly, so the community pathway
        is optimized through the ranking objective itself. A dot product
        (bounded in [0, 1]) is used instead of a log form: items without
        community support then receive a neutral 0 rather than a large
        negative constant.
        """
        graph = self.graph_projection(self.raw_graph_profiles)
        embeddings = F.normalize(graph[self.target_domain_id], dim=-1)
        logits = representation @ embeddings.T / self.adapter_config.community_temperature
        logits = logits.masked_fill(~self.community_mask[self.target_domain_id], -1e9)
        predicted = torch.softmax(logits, dim=-1)
        memberships = self.item_fine_memberships[self.target_domain_id, item_ids]
        return torch.einsum("bk,bnk->bn", predicted, memberships)

    def _all_item_representations(self) -> torch.Tensor:
        graph = self.graph_projection(self.raw_graph_profiles)
        graph_item = torch.einsum("dik,dkh->ih", self.item_fine_memberships, graph)
        items = self.item_embedding.weight + torch.tanh(self.graph_item_scale) * graph_item
        if self.adapter_config.use_item_content:
            projected = self.content_dropout_layer(
                self.content_projection(self.item_content)
            )
            items = items + torch.tanh(self.content_item_scale) * projected
        return items

    def _content_pool(self, items: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """Causal attention pooling of item content over the visible prefix.

        Query position q attends only to content of items at positions <= q,
        so each supervision point sees exactly the prefix the causal encoder
        sees. Padding ids carry a zero content row and contribute nothing.
        """
        projected_full = self.content_dropout_layer(
            self.content_projection(self.item_content)
        )
        projected = projected_full[items]
        pad = items.eq(0)
        length = queries.shape[1]
        causal = torch.tril(torch.ones(length, length, device=items.device, dtype=torch.bool))
        logits = torch.einsum(
            "bqh,blh->bql", queries, projected
        ) / float(queries.shape[-1]) ** 0.5
        allowed = causal.unsqueeze(0) & ~pad.unsqueeze(1)
        weights = torch.softmax(logits.masked_fill(~allowed, -1e9), dim=-1)
        # A query whose entire prefix is padding (user without target
        # history) would otherwise receive a uniform distribution and leak
        # future content; it must pool to exactly zero instead.
        weights = weights.masked_fill(~allowed.any(dim=-1, keepdim=True), 0.0)
        return torch.einsum("bql,blh->bqh", weights, projected)

    def _encode(self, items: torch.Tensor,
                all_items: torch.Tensor | None = None) -> torch.Tensor:
        if all_items is None:
            all_items = self._all_item_representations()
        return self.target_encoder(F.embedding(items, all_items), items.eq(0))

    def _target_sequence(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        rows = []
        length = self.sequence_length + 1
        for items, domains in zip(batch["mixed_item_seq"], batch["mixed_domain_seq"]):
            selected = items[domains.eq(self.target_domain_id + 1)]
            selected = selected[selected.ne(0)][-length:].tolist()
            rows.append([0] * (length - len(selected)) + selected)
        return torch.tensor(rows, dtype=torch.long, device=batch["mixed_item_seq"].device)

    def _affiliation(self, state: torch.Tensor, user_ids: torch.Tensor,
                     local_profiles: torch.Tensor) -> torch.Tensor:
        prior = self.memberships[self.target_domain_id, user_ids]
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
        logits = F.cosine_similarity(state.unsqueeze(1),
                                    local_profiles[self.target_domain_id].unsqueeze(0), dim=-1)
        logits = logits / self.adapter_config.community_temperature
        logits = logits.masked_fill(~self.community_mask[self.target_domain_id], -1e9)
        intent = torch.softmax(logits, dim=1)
        present = self.memberships[self.target_domain_id, user_ids].sum(dim=1, keepdim=True).gt(0)
        certainty = 1 + (intent * intent.clamp_min(1e-8).log()).sum(dim=1, keepdim=True) / max(
            float(torch.log(self.community_mask[self.target_domain_id].sum().clamp_min(2))), 1e-8
        )
        weight = torch.sigmoid(self.affiliation_gate(torch.cat((state, certainty), dim=1)))
        mixed = weight * prior + (1.0 - weight) * intent
        return torch.where(present, mixed, intent)

    def _local_target(self, state: torch.Tensor, user_ids: torch.Tensor,
                      visible_target_counts: torch.Tensor,
                      graph: torch.Tensor,
                      affiliation: torch.Tensor,
                      content_state: torch.Tensor | None = None,
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Target-local community context is structural, not LLM semantic
        # injection. LLM profiles are activated only through the bridge below.
        community = affiliation @ graph[self.target_domain_id]
        community = torch.tanh(self.community_scale) * community
        confidence = 1.0 - torch.exp(
            -visible_target_counts.float() / self.adapter_config.user_confidence_tau
        )
        user = confidence.unsqueeze(1) * self.target_user_embedding(user_ids)
        weights = torch.softmax(
            self.local_weight_net(torch.cat((state, user, community), dim=1)), dim=1
        )
        fused = (weights[:, 0:1] * state + weights[:, 1:2] * user
                 + weights[:, 2:3] * community)
        if content_state is not None:
            # Applied before the norm with a zero-initialized scale, so the
            # target-only branch starts exactly at the community model.
            fused = fused + torch.tanh(self.content_user_scale) * content_state
        return self.target_norm(fused), community, confidence

    def _community_transfer(
        self, target: torch.Tensor, target_community: torch.Tensor,
        affiliation: torch.Tensor, shared: torch.Tensor,
        behavior: torch.Tensor, graph: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        source_contexts, qualities, schemes = [], [], []
        for source in range(self.num_domains):
            if source == self.target_domain_id:
                continue
            row_reliability = self.bridge_reliability[self.target_domain_id, source]
            reliability = torch.einsum("bk,rk->br", affiliation, row_reliability)
            router = self.scheme_router(torch.cat((
                target, target_community, reliability,
            ), dim=1))
            if self.training and self.adapter_config.scheme_dropout > 0:
                dropped = torch.rand_like(router).lt(self.adapter_config.scheme_dropout)
                dropped[dropped.all(dim=1), 0] = False
                router = router.masked_fill(dropped, -1e4)
            if self.audit_disable_semantic_route:
                router = router.clone()
                router[:, 0] = -1e4
            scheme = torch.softmax(router, dim=1)
            if self.audit_bridge_scheme is not None:
                scheme = F.one_hot(
                    torch.full((len(target),), self.audit_bridge_scheme,
                               dtype=torch.long, device=target.device),
                    num_classes=3,
                ).to(target.dtype)
            mapped = torch.einsum(
                "bk,rkj->brj", affiliation,
                self.base_bridges[self.target_domain_id, source],
            )
            activated_mapping = torch.einsum("br,brj->bj", scheme, mapped)
            accepted_mass = activated_mapping.sum(dim=1, keepdim=True)
            conditional_mapping = activated_mapping / accepted_mass.clamp_min(1e-8)
            mapping_entropy = -(
                conditional_mapping.clamp_min(1e-8)
                * conditional_mapping.clamp_min(1e-8).log()
            ).sum(dim=1) / float(torch.log(self.community_mask[source].sum().clamp_min(2)))
            map_certainty = 1.0 - mapping_entropy
            scheme_entropy = -(
                scheme.clamp_min(1e-8) * scheme.clamp_min(1e-8).log()
            ).sum(dim=1) / float(torch.log(torch.tensor(3.0)))
            ot_confidence = (scheme * reliability).sum(dim=1)
            target_certainty = 1.0 - (
                -(affiliation.clamp_min(1e-8) * affiliation.clamp_min(1e-8).log()).sum(dim=1)
                / float(torch.log(self.community_mask[self.target_domain_id].sum().clamp_min(2)))
            )
            agreement = (
                F.cosine_similarity(mapped[:, 0], mapped[:, 1], dim=1)
                + F.cosine_similarity(mapped[:, 0], mapped[:, 2], dim=1)
                + F.cosine_similarity(mapped[:, 1], mapped[:, 2], dim=1)
            ) / 3.0
            # Query conditional rules INSIDE the candidate source communities.
            # No independently learned source graph coordinates are transported.
            keys = self.memory_key_projection(self.memory_keys[source])
            values = self.memory_value_projection(self.memory_values[source])
            support = self.memory_support[source]
            attention_logits = torch.einsum(
                "bh,kmh->bkm", F.normalize(target, dim=-1), F.normalize(keys, dim=-1)
            ) / self.adapter_config.community_temperature
            attention_logits = attention_logits + support.clamp_min(1e-8).log()
            weights = torch.softmax(attention_logits, dim=-1) * support.gt(0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            memory = torch.einsum("bkm,kmh->bkh", weights, values)
            evidence = torch.einsum("bkm,km->bk", weights, support)
            if self.audit_disable_profiles:
                memory = torch.zeros_like(memory)
            memory = memory + behavior[source].unsqueeze(0)
            source_contexts.append(torch.einsum(
                "bk,bk,bkh->bh", activated_mapping, evidence, memory
            ))
            qualities.append(torch.stack((ot_confidence, map_certainty,
                                          1.0 - scheme_entropy,
                                          target_certainty, agreement), dim=1))
            schemes.append(scheme)
        if not source_contexts:
            zeros = torch.zeros_like(target)
            quality = target.new_zeros((len(target), 5))
            scheme = target.new_zeros((len(target), 3))
        else:
            zeros = torch.stack(source_contexts).mean(dim=0)
            quality = torch.stack(qualities).mean(dim=0)
            scheme = torch.stack(schemes).mean(dim=0)
        reliability_features = quality.clamp(0, 1)
        bridge_quality = (reliability_features * target.new_tensor(
            [0.30, 0.25, 0.15, 0.15, 0.15]
        )).sum(dim=1, keepdim=True)
        content = torch.sigmoid(self.bridge_gate(torch.cat((
            target, zeros, target * zeros, reliability_features,
        ), dim=1)))
        gate = bridge_quality.detach() * content
        return zeros, {
            "scheme_weights": scheme,
            "reliability_features": reliability_features,
            "bridge_quality": bridge_quality,
            "content_gate": content,
            "transfer_gate": gate,
        }

    def _represent_states(self, state: torch.Tensor, user_ids: torch.Tensor,
                          visible_target_counts: torch.Tensor,
                          content_state: torch.Tensor | None = None,
                          ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        shared, behavior, graph = self._profiles()
        affiliation = self._affiliation(state, user_ids, graph)
        target_only, target_community, user_confidence = self._local_target(
            state, user_ids, visible_target_counts, graph, affiliation,
            content_state=content_state,
        )
        source, auxiliary = self._community_transfer(
            target_only, target_community, affiliation, shared, behavior, graph
        )
        warming_up = self.training and int(self.training_epoch) < self.adapter_config.backbone_warmup_epochs
        final = target_only if self.audit_disable_transfer or warming_up else (
            target_only + self.adapter_config.transfer_scale
            * torch.tanh(self.residual_scale) * auxiliary["transfer_gate"] * source
        )
        auxiliary["target_affiliation"] = affiliation
        auxiliary["user_confidence"] = user_confidence
        auxiliary["source_context"] = source
        return final, target_only, auxiliary

    def _represent(self, batch: Mapping[str, torch.Tensor],
                   all_items: torch.Tensor | None = None,
                   ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        items = self._target_sequence(batch)
        encoded = self._encode(items, all_items)
        state = encoded[:, -1]
        visible = items.ne(0).sum(dim=1)
        content_state = None
        if self.adapter_config.use_item_content:
            content_state = self._content_pool(items, encoded)[:, -1]
        return self._represent_states(state, batch["user_id"], visible,
                                      content_state=content_state)

    def _score(self, representation: torch.Tensor, item_ids: torch.Tensor,
               all_items: torch.Tensor | None = None) -> torch.Tensor:
        if all_items is None:
            all_items = self._all_item_representations()
        embeddings = F.embedding(item_ids, all_items)
        return torch.einsum("bd,bnd->bn", representation, embeddings) + self.item_bias(item_ids).squeeze(-1)

    def calculate_loss(self, batch: Mapping[str, torch.Tensor]) -> LossOutput:
        all_items = self._all_item_representations()
        encoded = self._encode(batch["comm_context"], all_items)[:, :-1]
        valid = batch["comm_positive"].ne(0)
        rows, positions = valid.nonzero(as_tuple=True)
        if not len(rows):
            raise ValueError("CommCDSR batch has no target-domain causal supervision")
        content_state = None
        if self.adapter_config.use_item_content:
            pooled = self._content_pool(batch["comm_context"][:, :-1], encoded)
            content_state = pooled[rows, positions]
        representation, target_only, auxiliary = self._represent_states(
            encoded[rows, positions], batch["comm_user_id"][rows],
            batch["comm_context"][:, :-1].ne(0).cumsum(dim=1)[rows, positions],
            content_state=content_state,
        )
        positive = batch["comm_positive"][rows, positions].unsqueeze(1)
        negative = batch["comm_negative"][rows, positions]
        mask = batch["comm_negative_mask"][rows, positions]
        positive_score = self._score(representation, positive, all_items)
        negative_score = self._score(representation, negative, all_items)
        score_weight = self.adapter_config.community_score_weight
        if score_weight > 0:
            # Ramp the score-level term in only after the target-only warm-up,
            # so an uncalibrated community prediction cannot distort the base
            # ranking while the backbone is still forming.
            started = int(self.training_epoch) - self.adapter_config.backbone_warmup_epochs
            ramp = max(self.adapter_config.community_score_ramp_epochs, 1)
            score_weight = score_weight * min(max(started / ramp, 0.0), 1.0)
        community_score_margin = positive.new_zeros(())
        if score_weight > 0:
            positive_community = self._community_score(representation, positive)
            negative_community = self._community_score(representation, negative)
            positive_score = positive_score + score_weight * positive_community
            negative_score = negative_score + score_weight * negative_community
            community_score_margin = (
                positive_community.mean()
                - (negative_community * mask.float()).sum(1).div(mask.float().sum(1).clamp_min(1)).mean()
            )
        target_positive = self._score(target_only, positive, all_items)
        target_negative = self._score(target_only, negative, all_items)
        def sampled_softmax(positive_values: torch.Tensor,
                            negative_values: torch.Tensor) -> torch.Tensor:
            negative_values = negative_values.masked_fill(~mask, -1e9)
            logits = torch.cat((positive_values, negative_values), dim=1)
            return -F.log_softmax(logits, dim=1)[:, 0]

        recommendation_rows = sampled_softmax(positive_score, negative_score)
        target_rows = sampled_softmax(target_positive, target_negative)
        recommendation, target_loss = recommendation_rows.mean(), target_rows.mean()
        safe = F.relu(recommendation_rows - target_rows.detach()
                      - self.adapter_config.safe_transfer_tolerance).mean()
        diversity = self._interest_diversity()
        loss = (recommendation + self.adapter_config.target_only_loss_weight * target_loss
                + self.adapter_config.safe_transfer_weight * safe
                + self.adapter_config.interest_diversity_weight * diversity)
        community_next = transfer_next = positive.new_zeros(())
        community_coverage = transfer_coverage = positive.new_zeros((), dtype=torch.float)
        if self.adapter_config.community_next_item_weight > 0:
            community_next, community_coverage = self._community_next_item_loss(
                representation, positive.squeeze(1)
            )
            loss = loss + self.adapter_config.community_next_item_weight * community_next
        if self.adapter_config.transfer_next_item_weight > 0:
            transfer_next, transfer_coverage = self._community_next_item_loss(
                auxiliary["source_context"], positive.squeeze(1)
            )
            loss = loss + self.adapter_config.transfer_next_item_weight * transfer_next
        scheme_entropy = -(
            auxiliary["scheme_weights"].clamp_min(1e-8)
            * auxiliary["scheme_weights"].clamp_min(1e-8).log()
        ).sum(dim=1).mean()
        outputs = {
            "loss": loss, "recommendation": recommendation,
            "target_only_recommendation": target_loss,
            "safe_transfer": safe,
            "residual_scale": torch.tanh(self.residual_scale),
            "transfer_gate": auxiliary["transfer_gate"].mean(),
            "content_gate": auxiliary["content_gate"].mean(),
            "bridge_quality": auxiliary["bridge_quality"].mean(),
            "bridge_scheme_entropy": scheme_entropy,
            "graph_item_scale": torch.tanh(self.graph_item_scale),
            "community_scale": torch.tanh(self.community_scale),
            "interest_diversity": diversity,
            "community_next_item": community_next,
            "community_next_item_coverage": community_coverage,
            "transfer_next_item": transfer_next,
            "community_score_margin": community_score_margin,
            "user_confidence": auxiliary["user_confidence"].mean(),
        }
        if self.adapter_config.use_item_content:
            outputs["content_user_scale"] = torch.tanh(self.content_user_scale)
            outputs["content_item_scale"] = torch.tanh(self.content_item_scale)
        return outputs

    def score_candidates(self, batch: Mapping[str, torch.Tensor],
                         candidate_ids: torch.Tensor) -> torch.Tensor:
        all_items = self._all_item_representations()
        representation, _, _ = self._represent(batch, all_items)
        scores = self._score(representation, candidate_ids, all_items)
        # Inference must see the same community-compatibility term the fused
        # branch was trained with; the target-only reference branch stays pure.
        if self.adapter_config.community_score_weight > 0:
            scores = scores + self.adapter_config.community_score_weight * self._community_score(
                representation, candidate_ids
            )
        return scores

    def score_target_only_candidates(self, batch: Mapping[str, torch.Tensor],
                                     candidate_ids: torch.Tensor) -> torch.Tensor:
        all_items = self._all_item_representations()
        _, target_only, _ = self._represent(batch, all_items)
        return self._score(target_only, candidate_ids, all_items)
