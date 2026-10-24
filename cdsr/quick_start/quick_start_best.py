"""End-to-end quick-start API for unified CDSR experiments."""

from __future__ import annotations

import json
import logging
import random
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, RandomSampler

from cdsr.configuration import load_experiment_config, prepare_experiment_data
from cdsr.data import (
    ABXITrainingDataset,
    AMIDTrainingDataset,
    C2DSRTrainingDataset,
    CommCDSRCollator,
    CommCDSRTrainingDataset,
    TriCDRCollator,
    TriCDRTrainingDataset,
    SyNCRecCollator,
    SyNCRecTrainingDataset,
    AutoCDSRCollator,
    AutoCDSRTrainingDataset,
    CMGRCollator,
    CMGRTrainingDataset,
    HGTLStepCollator,
    HGTLStepDataset,
    HorizonRecCollator,
    HorizonRecTrainingDataset,
    LLM4CDSRCollator,
    LLM4CDSRTrainingDataset,
    SemaCDRStepCollator,
    SemaCDRStepDataset,
    CanonicalCDSRDataset,
    collate_abxi_training,
    collate_amid_training,
    collate_c2dsr_training,
    collate_cdsr,
    load_corpus,
)
from cdsr.evaluation import CDSREvaluator, EvaluationConfig
from cdsr.features.comm_cdsr_best import (
    prepare_comm_cdsr_features,
    resolve_num_communities,
)
from cdsr.models.comm_cdsr_best import CommCDSRConfig, CommCDSRModel
from cdsr.protocol import load_protocol
from cdsr.protocol.adapters import export_adapter_data
from cdsr.protocol.bundle import ProtocolBundle, validate_protocol_against_corpus
from cdsr.trainers import StandardTrainer, TrainerConfig


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_logger(
    project_root: Path, model: str, log_dir: str | Path | None = None
) -> tuple[logging.Logger, Path]:
    log_dir = Path(log_dir).resolve() if log_dir is not None else project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"run_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = log_dir / f"{run_name}.log"
    logger = logging.getLogger(f"cdsr.{run_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s    %(levelname)s  %(message)s", datefmt="%d %b %H:%M"
    )
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, log_file


def _log_section(logger: logging.Logger, title: str, values: Mapping[str, Any]) -> None:
    body = "\n".join(f"{key} = {value}" for key, value in values.items())
    logger.info("\n%s:\n%s", title, body)


def _format_metrics(metrics: Mapping[str, float]) -> str:
    return "    ".join(
        f"{key} : {value:.4f}" for key, value in metrics.items() if key != "count"
    )


def _best_validation_result(fit_result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the validation payload associated with the restored checkpoint."""

    best_epoch = fit_result.get("best_epoch")
    candidates = [
        record for record in fit_result.get("history", [])
        if isinstance(record, Mapping) and isinstance(record.get("valid"), Mapping)
    ]
    if best_epoch is not None:
        for record in candidates:
            # Trainers log human-facing epochs as zero based while history is one based.
            if int(record.get("epoch", -1)) - 1 == int(best_epoch):
                return record["valid"]
    return candidates[-1]["valid"] if candidates else None


def _evaluation_examples(corpus, split: str, target_domain_id: int):
    examples = [
        example
        for example in CanonicalCDSRDataset(corpus, split)
        if example.target_domain_id == target_domain_id
    ]
    if not examples:
        raise ValueError(f"No {split} examples found for target domain {target_domain_id}")
    return examples


class _LimitedLoader:
    """Expose only the first batches of a loader for an opt-in smoke test."""

    def __init__(self, loader: DataLoader, max_batches: int):
        if max_batches < 1:
            raise ValueError("smoke_test.max_train_batches must be positive")
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = loader.dataset

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for index, batch in enumerate(self.loader):
            if index >= self.max_batches:
                break
            yield batch

    def __len__(self) -> int:
        return min(len(self.loader), self.max_batches)


def run_cdsr(
    model: str | None = None,
    config_file_list: Sequence[str | Path] | None = None,
    config_dict: Mapping[str, Any] | None = None,
    saved: bool = True,
    reuse_prepared: bool = False,
    log_dir: str | Path | None = None,
    run_mode: str = "train_test",
) -> dict[str, Any]:
    """Train a model using the shared protocol, optionally withholding test.

    Args:
        model: Registered native baseline name.
        config_file_list: Overall, dataset, and model YAML paths, in that order.
        config_dict: Optional nested configuration overrides.
        saved: Whether to save the trained model checkpoint.
        reuse_prepared: Load an already-built canonical corpus and protocol.
        log_dir: Optional directory for this run's framework log.
        run_mode: ``train_test`` for a final run or ``tune`` for validation only.
    """

    if model is None:
        raise ValueError("model is required")
    if run_mode not in {"train_test", "tune"}:
        raise ValueError("run_mode must be 'train_test' or 'tune'")
    if config_file_list is None or len(config_file_list) != 3:
        raise ValueError(
            "config_file_list must contain overall, dataset, and model YAML files in that order"
        )
    if model != "comm_cdsr_best":
        raise ValueError(f"This anonymous release ships only comm_cdsr_best (got {model!r})")

    overall, dataset_config, model_config = [Path(path).resolve() for path in config_file_list]
    project_root = overall.parent.parent
    logger, log_file = _init_logger(project_root, model, log_dir)
    config = load_experiment_config(
        overall=overall,
        dataset=dataset_config,
        model=model_config,
        project_root=project_root,
        overrides=config_dict,
    )
    configured_model = str(config["model"]["name"])
    if configured_model != model:
        raise ValueError(
            f"Requested model {model!r} does not match model YAML value {configured_model!r}"
        )

    training = config["training"]
    evaluation = config["evaluation"]
    protocol = config["protocol"]
    parameters = config["model"].get("parameters", {})
    target_only = bool(parameters.get("target_only", False))
    smoke_settings = config.get("smoke_test", {})
    smoke_enabled = bool(smoke_settings.get("enabled", False))
    smoke_train_batches = int(smoke_settings.get("max_train_batches", 1))
    smoke_eval_examples = int(smoke_settings.get("max_eval_examples", 8))
    if smoke_enabled and smoke_eval_examples < 1:
        raise ValueError("smoke_test.max_eval_examples must be positive")
    requested_device = str(training.get("device", "cuda:0"))
    device = "cuda:0" if requested_device == "auto" else requested_device
    if not device.startswith("cuda"):
        raise ValueError(
            f"GPU execution is required for quick_start, but training.device is {device!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "training.device requests CUDA, but PyTorch cannot access a GPU. "
            "Check CUDA_VISIBLE_DEVICES, NVIDIA driver access, and the execution environment."
        )
    torch.cuda.set_device(torch.device(device))

    protocol_seed = int(protocol["seed"])
    seed = int(config["experiment"].get("run_seed", protocol_seed))
    _log_section(
        logger,
        "General Hyper Parameters",
        {
            "gpu_id": torch.device(device).index or 0,
            "use_gpu": True,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(torch.device(device)),
            "seed": seed,
            "protocol_seed": protocol_seed,
            "reproducibility": config["experiment"].get("reproducible", True),
            "model": model,
            "config_id": config.config_id,
            "run_mode": run_mode,
        },
    )
    _log_section(logger, "Training Hyper Parameters", training)
    _log_section(logger, "Evaluation Hyper Parameters", evaluation)
    if smoke_enabled:
        logger.warning(
            "SMOKE TEST MODE: at most %d training batch(es) and %d evaluation "
            "example(s); results are integration checks and MUST NOT be reported as metrics",
            smoke_train_batches,
            smoke_eval_examples,
        )
    _log_section(
        logger,
        "Dataset Hyper Parameters",
        {
            "dataset": config["dataset"]["name"],
            "source_domains": [entry["name"] for entry in config["data"]["source_domains"]],
            "target_domain": config["data"]["target_domain"]["name"],
            "split_strategy": config["data"]["split_strategy"],
            "candidate_mode": protocol["candidate_mode"],
            "num_negatives": protocol["num_negatives"],
        },
    )

    if reuse_prepared:
        preparation = {
            "canonical_dir": str(Path(config["paths"]["canonical_dir"]).resolve()),
            "protocol_dir": str(Path(config["paths"]["protocol_dir"]).resolve()),
        }
        logger.info("Reusing prebuilt canonical dataset and frozen evaluation protocol")
    else:
        logger.info("Preparing canonical dataset and frozen evaluation protocol")
        preparation = prepare_experiment_data(config)
    corpus = load_corpus(preparation["canonical_dir"])
    protocol_bundle = load_protocol(preparation["protocol_dir"])
    if not reuse_prepared:
        validate_protocol_against_corpus(protocol_bundle, corpus)
    native_multi_domain = {"autocdsr", "syncrec", "hgtl"}
    if model in native_multi_domain and corpus.num_domains < 3:
        raise ValueError(
            f"{model} is configured as a native multi-domain baseline and requires "
            "Books + CD -> Movies; use amazon_books_cd_movies*.yaml"
        )
    if model == "comm_cdsr_best" and corpus.num_domains < 2:
        raise ValueError("comm_cdsr_best requires at least one source and one target domain")
    if model not in native_multi_domain | {"comm_cdsr_best"} and corpus.num_domains != 2:
        raise ValueError(
            f"{model} is a two-domain baseline and requires Books -> Movies; "
            "use amazon_books_movies*.yaml"
        )
    adapter_export = export_adapter_data(
        model, corpus, protocol_bundle, config["paths"]["adapter_dir"],
        validate_protocol=not reuse_prepared,
    )
    logger.info(
        "Dataset: users=%d, items=%d, domains=%d, train=%d, valid=%d, test=%d",
        corpus.num_users,
        corpus.num_items,
        corpus.num_domains,
        len(corpus.examples["train"]),
        len(corpus.examples["valid"]),
        len(corpus.examples["test"]),
    )
    logger.info("User groups: %s", corpus.metadata.get("user_group_counts", {}))
    _seed_everything(seed)
    target_domain_name = str(config["data"]["target_domain"]["name"])
    target_domain_id = corpus.domain_to_id[target_domain_name]
    observed_user_ids: set[int] | None = None
    if model == "amid":
        overlap_users = sorted(
            user_id for user_id, group in corpus.user_groups.items() if group == "overlap"
        )
        non_overlap_users = sorted(
            user_id for user_id, group in corpus.user_groups.items() if group != "overlap"
        )
        observed_ratio = float(parameters.get("observed_non_overlap_ratio", 0.25))
        observed_count = int(len(non_overlap_users) * observed_ratio)
        observed_non_overlap = random.Random(seed).sample(non_overlap_users, observed_count)
        observed_user_ids = set(overlap_users).union(observed_non_overlap)
        logger.info(
            "AMID observation view: overlap=%d, observed_non_overlap=%d, unobserved_non_overlap=%d",
            len(overlap_users),
            len(observed_non_overlap),
            len(non_overlap_users) - len(observed_non_overlap),
        )

    if model == "abxi":
        if target_only and target_domain_id != 1:
            raise ValueError("The current ABXI target-only adapter expects target domain 1")
        cdsr_model = ABXIModel(
            corpus,
            ABXIConfig(
                embedding_size=int(parameters.get("d_embed", 256)),
                num_heads=int(parameters.get("n_head", 2)),
                dropout=float(parameters.get("dropout", 0.5)),
                temperature=float(parameters.get("temp", 0.75)),
                domain_lora_rank=int(parameters.get("rd", 8)),
                invariant_lora_rank=int(parameters.get("ri", 8)),
                negatives_per_domain=int(parameters.get("n_neg", 128)),
                target_only=target_only,
            ),
        )
    elif model == "amid":
        cdsr_model = AMIDModel(
            corpus,
            AMIDConfig(
                embedding_size=int(parameters.get("emb_dim", 128)),
                hidden_size=int(parameters.get("hid_dim", 32)),
                sequence_length=int(parameters.get("seq_len", 50)),
                group_batch_size=int(training["batch_size"]),
                long_length=int(parameters.get("long_length", 7)),
                use_inner_compensation=bool(parameters.get("is_inc", False)),
                use_inter_compensation=bool(parameters.get("is_itc", True)),
                inner_threshold=float(parameters.get("ts1", 0.5)),
                inter_threshold=float(parameters.get("ts2", 0.4)),
                use_doubly_robust=bool(parameters.get("is_dr", True)),
                dr_estimation_weight=float(parameters.get("dr_e_weight", 0.1)),
                observed_user_ids=tuple(sorted(observed_user_ids or ())),
                target_only=target_only,
            ),
        )
    elif model == "c2dsr":
        cdsr_model = C2DSRModel(
            corpus,
            C2DSRConfig(
                hidden_units=int(parameters.get("hidden_units", 256)),
                num_blocks=int(parameters.get("num_blocks", 2)),
                num_heads=int(parameters.get("num_heads", 1)),
                gnn_depth=int(parameters.get("gnn_depth", 1)),
                dropout=float(parameters.get("dropout", 0.2)),
                leaky_relu=float(parameters.get("leaky_relu", 0.1)),
                sequence_length=int(parameters.get("maxlen", 15)),
                recommendation_weight=float(parameters.get("lambda", 0.7)),
                target_only=target_only,
            ),
        )
    elif model == "tricdr":
        cdsr_model = TriCDRModel(
            corpus,
            TriCDRConfig(
                hidden_units=int(parameters.get("hidden_units", 64)),
                num_blocks=int(parameters.get("num_blocks", 2)),
                num_heads=int(parameters.get("num_heads", 1)),
                dropout_rate=float(parameters.get("dropout_rate", 0.2)),
                sequence_length=int(parameters.get("maxlen", 200)),
                temperature=float(parameters.get("temperature", 5.0)),
                info_nce_temperature=float(parameters.get("info_nce_temperature", 0.1)),
                rate_mix_source=float(parameters.get("rate_mix_source", 1.0)),
                rate_mix_target=float(parameters.get("rate_mix_target", 1.0)),
                rate_source_target=float(parameters.get("rate_source_target", 1.0)),
                cl_weight=float(parameters.get("cl_weight", 1.0)),
                triplet_weight=float(parameters.get("triplet_weight", 1.0)),
                triplet_margin=float(parameters.get("triplet_margin", 1.0)),
                target_only=target_only,
            ),
            target_domain_id=target_domain_id,
        )
    elif model == "syncrec":
        cdsr_model = SyNCRecModel(
            corpus,
            SyNCRecConfig(
                hidden_size=int(parameters.get("hidden_size", 128)),
                num_hidden_layers=int(parameters.get("num_hidden_layers", 2)),
                num_attention_heads=int(parameters.get("num_attention_heads", 2)),
                hidden_act=str(parameters.get("hidden_act", "gelu")),
                attention_dropout=float(parameters.get("attention_dropout", 0.5)),
                hidden_dropout=float(parameters.get("hidden_dropout", 0.5)),
                initializer_range=float(parameters.get("initializer_range", 0.02)),
                sequence_length=int(parameters.get("max_seq_length", 50)),
                temperature=float(parameters.get("temperature", 1.0)),
                cross_expert_ratio=float(parameters.get("cross_expert_ratio", 0.8)),
                cross_detach=bool(parameters.get("cross_detach", False)),
                single_detach=bool(parameters.get("single_detach", False)),
                expert_layer=str(parameters.get("expert_layer", "transformer")),
                expert_num=int(parameters.get("expert_num", 5)),
                use_loss_rebalancing=bool(parameters.get("lrb", True)),
                use_mip=bool(parameters.get("mip", True)),
            ),
        )
    elif model == "autocdsr":
        from cdsr.models.autocdsr import AutoCDSRConfig, AutoCDSRModel

        cdsr_model = AutoCDSRModel(
            corpus,
            AutoCDSRConfig(
                embedding_size=int(parameters.get("d_model", 32)),
                feed_forward_size=int(parameters.get("d_ff", 32)),
                key_value_size=int(parameters.get("d_kv", 32)),
                general_heads=int(parameters.get("general_heads", 4)),
                domain_heads=int(parameters.get("domain_heads", 2)),
                general_layers=int(parameters.get("general_layers", 3)),
                domain_layers=int(parameters.get("domain_layers", 3)),
                dropout=float(parameters.get("dropout_rate", 0.15)),
                sequence_length=int(parameters.get("sequence_length", 800)),
                num_ib_tokens=int(parameters.get("num_ib_tokens", 4)),
                ib_comm_layers=tuple(
                    map(int, parameters.get("ib_comm_layers", [0, 1, 2]))
                ),
                contrastive_temperature=float(parameters.get("contrastive_tau", 0.1)),
            ),
        )
    elif model == "cmgr":
        use_cmgr_features = bool(parameters.get("global_emb", False)) or bool(
            parameters.get("local_emb", False)
        )
        cmgr_features = (
            prepare_cmgr_features(
                corpus,
                config.get("cmgr_features", {}),
                hidden_size=int(parameters.get("hidden_size", 128)),
            )
            if use_cmgr_features
            else None
        )
        if cmgr_features is not None:
            logger.info(
                "CMGR frozen catalog-aligned embeddings: fingerprint=%s, "
                "local_A=%s, local_B=%s, global=%s",
                cmgr_features.manifest["fingerprint"],
                tuple(cmgr_features.local_a.shape),
                tuple(cmgr_features.local_b.shape),
                tuple(cmgr_features.global_all.shape),
            )
        cdsr_model = CMGRModel(
            corpus,
            CMGRConfig(
                aspects=int(parameters.get("aspects", 200)),
                interests=int(parameters.get("interests", 10)),
                hidden_size=int(parameters.get("hidden_size", 128)),
                transformer_layers=int(parameters.get("transformer_layers", 2)),
                num_heads=int(parameters.get("num_heads", 1)),
                dropout_rate=float(parameters.get("dropout_rate", 0.5)),
                sequence_length=int(parameters.get("max_len", 200)),
                gamma=float(parameters.get("gamma", 1.0)),
                global_emb=bool(parameters.get("global_emb", False)),
                local_emb=bool(parameters.get("local_emb", False)),
                freeze_emb=bool(parameters.get("freeze_emb", False)),
                target_only=target_only,
            ),
            features=cmgr_features,
        )
    elif model == "hgtl":
        item_files = {
            str(entry["name"]): str(entry["item_file"])
            for entry in [*config["data"]["source_domains"], config["data"]["target_domain"]]
        }
        cdsr_model = HGTLModel(
            corpus,
            item_files,
            HGTLConfig(
                hidden_units=int(parameters.get("hidden_units", 50)),
                num_blocks=int(parameters.get("num_blocks", 2)),
                num_heads=int(parameters.get("num_heads", 1)),
                dropout_rate=float(parameters.get("dropout_rate", 0.5)),
                sequence_length=int(parameters.get("maxlen", 100)),
                contrastive_weight=float(parameters.get("contrastive_weight", 0.5)),
                l2_embedding=float(parameters.get("l2_embedding", 0.001)),
                max_category_length=int(parameters.get("max_category_length", 6)),
                max_neighbor_length=int(parameters.get("max_neighbor_length", 10)),
            ),
            device=device,
        )
    elif model == "horizonrec":
        target_domain_name = str(config["data"]["target_domain"]["name"])
        target_domain_id = corpus.domain_to_id[target_domain_name]
        horizon_architecture = {
            "hidden_units": int(parameters.get("hidden_units", 64)),
            "num_blocks": int(parameters.get("num_blocks", 2)),
            "num_heads": int(parameters.get("num_heads", 1)),
            "dropout_rate": float(parameters.get("dropout_rate", 0.2)),
            "sequence_length": int(parameters.get("max_len", 200)),
        }
        use_pretraining = bool(parameters.get("pretrain", False))
        horizon_pretraining = None
        if use_pretraining:
            horizon_pretraining = prepare_horizonrec_pretraining(
                corpus,
                target_domain_id,
                config.get("horizon_pretraining", {}),
                horizon_architecture,
                seed=seed,
                device=device,
                logger=logger,
            )
            logger.info(
                "HorizonRec canonical pretraining: fingerprint=%s, cache=%s",
                horizon_pretraining.manifest["fingerprint"],
                horizon_pretraining.cache_dir,
            )
        else:
            logger.warning(
                "HorizonRec variant: released Toy/Game and Douban pretraining is not "
                "catalog-aligned; running without released pretraining"
            )
        cdsr_model = HorizonRecModel(
            corpus,
            target_domain_id,
            HorizonRecConfig(
                hidden_units=horizon_architecture["hidden_units"],
                num_blocks=horizon_architecture["num_blocks"],
                num_heads=horizon_architecture["num_heads"],
                dropout_rate=horizon_architecture["dropout_rate"],
                sequence_length=horizon_architecture["sequence_length"],
                timesteps=int(parameters.get("timesteps", 32)),
                beta_start=float(parameters.get("beta_start", 0.001)),
                beta_end=float(parameters.get("beta_end", 0.02)),
                beta_schedule=str(parameters.get("beta_schedule", "cosine")),
                temperature=float(parameters.get("temperature", 1.0)),
                guidance_weight=float(parameters.get("guidance_weight", 0.6)),
                diffusion_loss_weight=float(parameters.get("diffusion_loss_weight", 0.1)),
                pretrain=bool(parameters.get("pretrain", False)),
                target_only=target_only,
            ),
            pretraining=horizon_pretraining,
        )
    elif model == "llm4cdsr":
        llm_train_view = LLM4CDSRTrainingDataset(
            corpus, int(parameters.get("max_len", 200))
        )
        entries = [*config["data"]["source_domains"], config["data"]["target_domain"]]
        entries_by_domain = sorted(
            entries, key=lambda entry: corpus.domain_to_id[str(entry["name"])]
        )
        llm_settings = config["llm"]
        llm_features = prepare_llm4cdsr_features(
            corpus,
            domain_entries=entries_by_domain,
            profile_user_ids=[record.user_id for record in llm_train_view.records],
            cache_root=llm_settings["cache_dir"],
            settings=llm_settings,
            hidden_size=int(parameters.get("hidden_size", 128)),
            seed=protocol_seed,
            logger=logger,
        )
        logger.info(
            "LLM4CDSR side features: source=%s, item_A=%s, item_B=%s, item_all=%s, users=%s",
            llm_features.source,
            tuple(llm_features.item_a.shape),
            tuple(llm_features.item_b.shape),
            tuple(llm_features.item_all.shape),
            tuple(llm_features.users.shape),
        )
        logger.info(
            "LLM4CDSR token usage: this_run=%s, cache_cumulative=%s",
            dict(llm_features.api_usage), dict(llm_features.api_usage_cumulative),
        )
        cdsr_model = LLM4CDSRModel(
            corpus,
            llm_features,
            LLM4CDSRConfig(
                hidden_size=int(parameters.get("hidden_size", 128)),
                transformer_layers=int(parameters.get("transformer_layers", 2)),
                num_heads=int(parameters.get("num_heads", 1)),
                dropout_rate=float(parameters.get("dropout_rate", 0.5)),
                sequence_length=int(parameters.get("max_len", 200)),
                tau=float(parameters.get("tau", 1.0)),
                tau_reg=float(parameters.get("tau_reg", 1.0)),
                alpha=float(parameters.get("alpha", 0.1)),
                beta=float(parameters.get("beta", 0.1)),
                local_emb=bool(parameters.get("local_emb", True)),
                global_emb=bool(parameters.get("global_emb", True)),
                freeze_emb=bool(parameters.get("freeze_emb", True)),
                target_only=target_only,
            ),
        )
    elif model == "comm_cdsr_best":
        entries = sorted(
            [*config["data"]["source_domains"], config["data"]["target_domain"]],
            key=lambda entry: corpus.domain_to_id[str(entry["name"])],
        )
        adaptive_gnn = str(config["community_memory"].get(
            "community_assignment", ""
        )) == "gnn_modularity"
        if adaptive_gnn:
            resolved_communities, community_sizing = None, {
                "mode": "learned_gnn_modularity",
                "selection": "Louvain modularity over train-only learned user-item graph",
            }
        else:
            resolved_communities, community_sizing = resolve_num_communities(
                corpus,
                parameters.get("num_communities", 8),
                config["community_memory"],
            )
        community_prep_started = time.perf_counter()
        comm_features = prepare_comm_cdsr_features(
            corpus,
            entries,
            config["community_memory"],
            communities=resolved_communities,
            seed=protocol_seed,
            logger=logger,
        )
        community_prep_seconds = time.perf_counter() - community_prep_started
        resolved_communities = int(comm_features.memberships.shape[-1])
        community_sizing.update({
            "resolved_communities": resolved_communities,
            "resolved_coarse_communities": int(comm_features.fine_to_coarse.shape[-1]),
            "community_sizes_by_domain": comm_features.manifest.get(
                "community_sizes_by_domain"
            ),
        })
        logger.info("CommCDSR community sizing: %s", community_sizing)
        # Item content embeddings live in their own cache, independent of the
        # community fingerprint, and consume no torch RNG.
        item_content = None
        content_prep_seconds = 0.0
        if bool(parameters.get("use_item_content", False)):
            from cdsr.features.comm_content_embeddings import (
                prepare_item_content_embeddings,
            )
            content_entries = sorted(
                [*config["data"]["source_domains"], config["data"]["target_domain"]],
                key=lambda entry: corpus.domain_to_id[str(entry["name"])],
            )
            content_prep_started = time.perf_counter()
            item_content = prepare_item_content_embeddings(
                corpus, content_entries, target_domain_id,
                config["community_memory"], logger=logger,
            )
            content_prep_seconds = time.perf_counter() - content_prep_started
        # Structured preparation timing (seconds), written next to the
        # community cache so later experiments can quote offline costs
        # without re-running preparation. Cache-hit runs still record the
        # (much shorter) load time; fresh builds record the full cost.
        try:
            _timing_path = Path(comm_features.cache_dir) / "preparation_timing.json"
            _timing_payload = {
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "dataset": str(config["data"].get("name", "")),
                "community_preparation_seconds": round(community_prep_seconds, 3),
                "content_preparation_seconds": round(
                    content_prep_seconds if item_content is not None else 0.0, 3
                ),
                "content_preparation_attempted": item_content is not None,
            }
            _timing_path.write_text(
                json.dumps(_timing_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            logger.info("Preparation timing: %s", _timing_payload)
        except Exception as error:  # Timing bookkeeping must never kill a run.
            logger.warning("Failed to write preparation timing: %s", error)
        # Feature preparation (formation GNN training, profile generation)
        # consumes the global RNG streams and does so only on a cache miss;
        # reseed here so the training trajectory is identical whether the
        # community memory was freshly built or loaded from cache.
        _seed_everything(seed)
        cdsr_model = CommCDSRModel(
            corpus,
            target_domain_id,
            comm_features,
            CommCDSRConfig(
                hidden_size=int(parameters.get("hidden_size", 64)),
                sequence_length=int(parameters.get("sequence_length", 100)),
                num_blocks=int(parameters.get("num_blocks", 2)),
                num_heads=int(parameters.get("num_heads", 1)),
                num_negatives=int(parameters.get("num_negatives", 64)),
                num_communities=resolved_communities,
                graph_feature_dim=int(comm_features.fine_graph_profiles.shape[-1]),
                dropout=float(parameters.get("dropout", 0.2)),
                community_temperature=float(parameters.get("community_temperature", 0.5)),
                transfer_scale=float(parameters.get("transfer_scale", 1.0)),
                ot_epsilon=float(parameters.get("ot_epsilon", 0.1)),
                ot_iterations=int(parameters.get("ot_iterations", 50)),
                community_mass_power=float(parameters.get("community_mass_power", 0.5)),
                scheme_dropout=float(parameters.get("scheme_dropout", 0.1)),
                unmatched_cost=float(parameters.get("unmatched_cost", 0.6)),
                safe_transfer_weight=float(parameters.get("safe_transfer_weight", 0.1)),
                safe_transfer_tolerance=float(parameters.get("safe_transfer_tolerance", 0.02)),
                backbone_warmup_epochs=int(parameters.get("backbone_warmup_epochs", 5)),
                target_only_loss_weight=float(parameters.get("target_only_loss_weight", 0.5)),
                user_confidence_tau=float(parameters.get("user_confidence_tau", 5.0)),
                bridge_topk=int(parameters.get("bridge_topk", 4)),
                intent_prior_weight=float(parameters.get("intent_prior_weight", 0.35)),
                interest_diversity_weight=float(
                    parameters.get("interest_diversity_weight", 0.0)
                ),
                community_next_item_weight=float(
                    parameters.get("community_next_item_weight", 0.0)
                ),
                transfer_next_item_weight=float(
                    parameters.get("transfer_next_item_weight", 0.0)
                ),
                community_score_weight=float(
                    parameters.get("community_score_weight", 0.0)
                ),
                community_score_ramp_epochs=int(
                    parameters.get("community_score_ramp_epochs", 5)
                ),
                use_item_content=bool(parameters.get("use_item_content", False)),
                content_dropout=float(parameters.get("content_dropout", 0.0)),
            ),
            item_content=item_content,
        )
        # Retrained-ablation wiring: the audit switches are plain attributes,
        # so a config flag can pin one on for the entire training run (e.g.
        # the "w/o community bridge" arm retrains with transfer disabled).
        _ablation = dict(config.get("ablation") or {})
        if _ablation.get("disable_transfer"):
            cdsr_model.audit_disable_transfer = True
            logger.info("ABLATION retrained arm: audit_disable_transfer=True")
        if _ablation.get("disable_profiles"):
            cdsr_model.audit_disable_profiles = True
            logger.info("ABLATION retrained arm: audit_disable_profiles=True")
        if _ablation.get("bridge_scheme") is not None:
            cdsr_model.audit_bridge_scheme = int(_ablation["bridge_scheme"])
            logger.info("ABLATION retrained arm: bridge_scheme=%s", _ablation["bridge_scheme"])
        logger.info(
            "CommCDSR profiles source=%s cache=%s; formation=train_split_only; "
            "fine_memberships=%s coarse_memberships=%s hierarchy=%s "
            "module2=identity_free_sparse_ot(shared_semantics,behavior,graph)+intent_router+reliability_activation "
            "token_usage=%s",
            comm_features.source,
            comm_features.cache_dir,
            tuple(comm_features.memberships.shape),
            tuple(comm_features.coarse_memberships.shape),
            tuple(comm_features.fine_to_coarse.shape),
            dict(comm_features.api_usage),
        )
        logger.info(
            "CommCDSR community diagnostics: formation=%s sizes=%s separation=%s selection=%s",
            comm_features.manifest.get("formation"),
            comm_features.manifest.get("community_sizes_by_domain"),
            comm_features.manifest.get("graph_profile_separation"),
            comm_features.manifest.get("community_selection"),
        )
    else:
        target_domain_name = str(config["data"]["target_domain"]["name"])
        target_domain_id = corpus.domain_to_id[target_domain_name]
        semantic_features = prepare_semacdr_features(
            corpus, config.get("semantic_features", {}),
            common_dim=int(parameters.get("common_llm_embedding_dim", 768)),
            unique_dim=int(parameters.get("unique_semantic_embedding_dim", 768)),
            seed=protocol_seed,
            domain_entries=sorted(
                [*config["data"]["source_domains"], config["data"]["target_domain"]],
                key=lambda entry: corpus.domain_to_id[str(entry["name"])],
            ),
            logger=logger,
        )
        logger.info("SemaCDR semantic features: source=%s, common=%s, unique=%s",
                    semantic_features.source, tuple(semantic_features.common.shape),
                    tuple(semantic_features.unique.shape))
        logger.info(
            "SemaCDR token usage: this_run=%s, cache_cumulative=%s",
            dict(semantic_features.api_usage),
            dict(semantic_features.api_usage_cumulative),
        )
        cdsr_model = SemaCDRModel(
            corpus, target_domain_id, semantic_features,
            SemaCDRConfig(
                hidden_units=int(parameters.get("hidden_units", 64)),
                num_blocks=int(parameters.get("num_blocks", 2)),
                num_heads=int(parameters.get("num_heads", 1)),
                dropout=float(parameters.get("dropout", 0.5)),
                sequence_length=int(parameters.get("maxlen", 100)),
                common_dim=int(parameters.get("common_llm_embedding_dim", 768)),
                unique_dim=int(parameters.get("unique_semantic_embedding_dim", 768)),
                temperature=float(parameters.get("infonce_temperature", 0.07)),
                cl1_weight=float(parameters.get("cl1_weight", 0.1)),
                cl2_weight=float(parameters.get("cl2_weight", 0.1)),
                gnn_depth=int(parameters.get("gnn_depth", 1)),
                leaky_relu=float(parameters.get("leaky_relu", 0.1)),
                target_only=target_only,
            ),
        )
    logger.info("%s", cdsr_model)
    logger.info(
        "Trainable parameters: %d",
        sum(parameter.numel() for parameter in cdsr_model.parameters() if parameter.requires_grad),
    )

    target_domain_name = str(config["data"]["target_domain"]["name"])
    target_domain_id = corpus.domain_to_id[target_domain_name]
    if model == "abxi":
        main_train_examples = ABXITrainingDataset(corpus, cdsr_model.sequence_length)
        train_collate = collate_abxi_training
    elif model == "amid":
        main_train_examples = AMIDTrainingDataset(
            corpus,
            cdsr_model.sequence_length,
            included_user_ids=observed_user_ids,
        )
        train_collate = collate_amid_training
        if target_only:
            pad = corpus.num_items + 1
            main_train_examples.records = [
                record for record in main_train_examples.records
                if record.target_domain_id == target_domain_id
                and sum(item != pad for item in record.domain_sequence_1) > 0
            ]
    elif model == "c2dsr":
        main_train_examples = C2DSRTrainingDataset(
            corpus,
            cdsr_model.sequence_length,
            seed=seed,
            official_batch_size=int(training["batch_size"]),
        )
        train_collate = collate_c2dsr_training
        if target_only:
            main_train_examples.records = [
                record for record in main_train_examples.records if any(record.values[15])
            ]
    elif model == "tricdr":
        main_train_examples = TriCDRTrainingDataset(
            corpus, cdsr_model.sequence_length, target_domain_id=target_domain_id
        )
        train_collate = TriCDRCollator(seed)
    elif model == "syncrec":
        main_train_examples = SyNCRecTrainingDataset(corpus, cdsr_model.sequence_length)
        train_collate = SyNCRecCollator(seed)
    elif model == "autocdsr":
        main_train_examples = AutoCDSRTrainingDataset(corpus, cdsr_model.sequence_length)
        train_collate = AutoCDSRCollator(
            seed,
            masking_probability=float(parameters.get("masking_probability", 0.15)),
            masking_tolerance=float(parameters.get("masking_tolerance", 0.05)),
            max_trials_for_masking=int(parameters.get("max_trials_for_masking", 5)),
        )
    elif model == "cmgr":
        main_train_examples = CMGRTrainingDataset(corpus, cdsr_model.sequence_length)
        train_collate = CMGRCollator(seed)
        if target_only:
            main_train_examples.records = [
                record for record in main_train_examples.records
                if sum(domain == target_domain_id for domain in record.domains) >= 2
            ]
    elif model == "hgtl":
        main_train_examples = HGTLStepDataset(corpus, int(training["batch_size"]))
        train_collate = HGTLStepCollator(
            corpus,
            int(training["batch_size"]),
            cdsr_model.sequence_length,
            seed,
        )
    elif model == "horizonrec":
        main_train_examples = HorizonRecTrainingDataset(
            corpus, cdsr_model.sequence_length, target_domain_id
        )
        train_collate = HorizonRecCollator(cdsr_model.sequence_length)
    elif model == "llm4cdsr":
        if target_only:
            main_train_examples = CMGRTrainingDataset(corpus, cdsr_model.sequence_length)
            main_train_examples.records = [
                record for record in main_train_examples.records
                if sum(domain == target_domain_id for domain in record.domains) >= 2
            ]
            train_collate = CMGRCollator(seed)
        else:
            main_train_examples = llm_train_view
            train_collate = LLM4CDSRCollator(seed, llm_train_view.profile_index_by_user)
    elif model == "semacdr":
        main_train_examples = SemaCDRStepDataset(
            corpus, target_domain_id, int(training["batch_size"])
        )
        train_collate = SemaCDRStepCollator(
            corpus, target_domain_id, int(training["batch_size"]),
            cdsr_model.sequence_length, seed,
        )
    elif model == "comm_cdsr_best":
        main_train_examples = CommCDSRTrainingDataset(
            corpus, cdsr_model.sequence_length, target_domain_id
        )
        train_collate = CommCDSRCollator(
            corpus, target_domain_id, cdsr_model.adapter_config.num_negatives, seed
        )
    else:
        main_train_examples = CanonicalCDSRDataset(corpus, "train")
        train_collate = collate_cdsr
    if model == "tricdr":
        batch_size = int(training["batch_size"])
        raw_official_steps = main_train_examples.population_count // batch_size
        # Upstream assumes a large corpus and otherwise produces zero batches.
        # Keep its fixed replacement batch intact for tiny smoke-test corpora.
        official_steps = max(1, raw_official_steps)
        if raw_official_steps == 0:
            logger.info(
                "Tri-CDR tiny-corpus compatibility: overlap_users=%d < batch_size=%d; "
                "using one full replacement-sampled batch",
                main_train_examples.population_count,
                batch_size,
            )
        generator = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(
            main_train_examples,
            replacement=True,
            num_samples=official_steps * batch_size,
            generator=generator,
        )
        train_loader = DataLoader(
            main_train_examples,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            collate_fn=train_collate,
        )
    elif model in {"hgtl", "semacdr"}:
        train_loader = DataLoader(
            main_train_examples,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=train_collate,
        )
    else:
        train_loader = DataLoader(
            main_train_examples,
            batch_size=int(training["batch_size"]),
            shuffle=model != "c2dsr",
            num_workers=int(training.get("num_workers", 0)),
            collate_fn=train_collate,
            drop_last=model == "autocdsr",
        )
    eval_batch_size = int(evaluation["eval_batch_size"])
    if model in {"amid", "autocdsr", "horizonrec", "llm4cdsr", "semacdr"}:
        eval_batch_size = min(eval_batch_size, int(training["batch_size"]))
    if model == "horizonrec":
        eval_batch_size = min(eval_batch_size, int(parameters.get("test_batch_size", 128)))
    if model == "cmgr":
        # CMGR adapts 1024-d item features for every sampled candidate. Keep
        # the exact frozen candidate rows, but score fewer users at once.
        eval_batch_size = min(eval_batch_size, int(parameters.get("test_batch_size", 64)))
    valid_examples = _evaluation_examples(corpus, "valid", target_domain_id)
    evaluation_protocol_bundle = protocol_bundle
    if smoke_enabled:
        valid_examples = valid_examples[:smoke_eval_examples]
        evaluation_protocol_bundle = ProtocolBundle(
            path=protocol_bundle.path,
            manifest=dict(protocol_bundle.manifest),
            rows={
                "valid": protocol_bundle.rows["valid"][: len(valid_examples)],
            },
        )
    elif run_mode == "tune":
        # Keep the test candidate rows out of the evaluator entirely. This makes
        # accidental test evaluation impossible during hyperparameter search.
        evaluation_protocol_bundle = ProtocolBundle(
            path=protocol_bundle.path,
            manifest=dict(protocol_bundle.manifest),
            rows={"valid": protocol_bundle.rows["valid"]},
        )
    valid_loader = DataLoader(
        valid_examples,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate_cdsr,
    )
    test_loader = None
    if run_mode == "train_test":
        test_examples = _evaluation_examples(corpus, "test", target_domain_id)
        if smoke_enabled:
            test_examples = test_examples[:smoke_eval_examples]
            evaluation_protocol_bundle.rows["test"] = protocol_bundle.rows["test"][
                : len(test_examples)
            ]
        test_loader = DataLoader(
            test_examples,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=int(training.get("num_workers", 0)),
            collate_fn=collate_cdsr,
        )
    if len(train_loader.dataset) == 0:
        raise ValueError("Canonical training split is empty")
    dr_loader = None
    if model == "amid":
        if not target_only:
            dr_examples = AMIDTrainingDataset(corpus, cdsr_model.sequence_length)
            dr_loader = DataLoader(
                dr_examples,
                batch_size=int(training["batch_size"]),
                shuffle=True,
                num_workers=int(training.get("num_workers", 0)),
                collate_fn=collate_amid_training,
            )
    if smoke_enabled:
        train_loader = _LimitedLoader(train_loader, smoke_train_batches)
        if dr_loader is not None:
            dr_loader = _LimitedLoader(dr_loader, smoke_train_batches)
    logger.info(
        "%s canonical training view: main_rows=%d, dr_rows=%s, sequence_length=%d%s",
        model.upper(),
        len(train_loader.dataset),
        len(dr_loader.dataset) if dr_loader is not None else "n/a",
        cdsr_model.sequence_length,
        (
            f", fixed_itc_batch={int(training['batch_size'])}, remainder_completion=real_rows"
            if model == "amid"
            else (
                ", users=official_usable_overlap, graph_users=all, supervision=last_10_positions"
                if model == "c2dsr"
                else (
                    f", users=overlap_replacement, steps={len(train_loader)}, "
                    f"source_domain_id={cdsr_model.source_domain_id}, "
                    f"target_domain_id={cdsr_model.target_domain_id}, target_domain_only=true"
                    if model == "tricdr"
                    else (
                        f", users=all, domains={corpus.num_domains}, "
                        "tasks=per_domain_plus_cross, supervision=all_positions"
                        if model == "syncrec"
                        else (
                            f", users=all, domains={corpus.num_domains}, "
                            "masking=bert4rec_random_0.15, truncation=prefix, "
                            "sparse_id_offset=100, drop_last=true"
                            if model == "autocdsr"
                            else (
                                ", users=all_usable, views=mixed+A+B, supervision=all_next_items, "
                                "negative_sampling=official_with_replacement"
                                if model == "cmgr"
                                else (
                                    f", users=all_domain_eligible, real_domains={corpus.num_domains}, "
                                    "sampler=per-domain_replacement, graph=train_split_only"
                                    if model == "hgtl"
                                    else (
                                        ", users=target_transition_eligible, views=target+source+mixed, "
                                        "mdr_pool=canonical_train_detached, pretrain="
                                        + (
                                            "canonical_train_catalog_aligned"
                                            if bool(parameters.get("pretrain", False))
                                            else "false"
                                        )
                                        if model == "horizonrec"
                                        else (
                                            ", users=official_usable_overlap, views=mixed+A+B, "
                                            "llm_features=canonical_train_profiles"
                                            if model == "llm4cdsr"
                                            else (
                                                f", users=target_transition_eligible, fixed_replacement_batch={int(training['batch_size'])}, "
                                                "graphs=canonical_train_only, source_graph_users=target_intersection, target_ids_first=true"
                                                if model == "semacdr"
                                                else (
                                                    f", users=target_transition_eligible, target_domain={target_domain_name}, "
                                                    "encoder=graph_semantic_enhanced_target_causal_attention, "
                                                    "supervision=all_target_causal_positions_once, "
                                                    f"target_unseen_negatives_without_replacement={cdsr_model.adapter_config.num_negatives}, "
                                                    "target_personalization=sequence_plus_gated_target_community, "
                                                    "cross_domain_user_sequence=false, mixed_user_sequence=false, "
                                                    "communities=train_only_dual_view_gnn_louvain_hierarchy, "
                                                    "bridge=identity_free_interest_behavior_topology_hierarchical_ot, "
                                                    "activation=target_intent_query_over_source_community_tokens, "
                                                    "module3=reliability_gated_community_cross_attention, "
                                                    f"profiles={cdsr_model.memory_source}"
                                                    if model == "comm_cdsr_best"
                                                    else ", users=overlap_only, supervision=all_positions"
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
    )

    evaluator = CDSREvaluator(
        corpus.domain_catalogs,
        EvaluationConfig(
            topk=tuple(map(int, evaluation["topk"])),
            candidate_mode=str(protocol["candidate_mode"]),
            num_sampled_negatives=int(protocol["num_negatives"]),
            filter_history=bool(protocol["filter_history"]),
            seed=protocol_seed,
            record_per_user_accuracy=bool(
                evaluation.get("record_per_user_accuracy", False)
            ),
            per_user_splits=tuple(
                map(str, evaluation.get("per_user_splits", ["test"]))
            ),
        ),
        protocol_bundle=evaluation_protocol_bundle,
    )
    if model == "abxi":
        trainer = ABXITrainer(
            cdsr_model,
            ABXITrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
                warmup_epochs=int(training.get("warmup_epochs", 1)),
                lr_gamma=float(training.get("lr_gamma", 0.3162)),
                lr_patience=int(training.get("lr_patience", 30)),
                early_stopping_patience=int(training["patience"]),
                grad_clip_norm=training.get("grad_clip_norm"),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
            ),
            logger=logger,
        )
    elif model == "amid":
        trainer = AMIDTrainer(
            cdsr_model,
            AMIDTrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                secondary_learning_rate_multiplier=float(
                    training.get("secondary_learning_rate", 0.01)
                ),
                weight_decay=float(training.get("weight_decay", 0.0)),
                grad_clip_norm=training.get("grad_clip_norm"),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
                patience=int(training["patience"]),
            ),
            logger=logger,
        )
    elif model == "c2dsr":
        trainer = C2DSRTrainer(
            cdsr_model,
            C2DSRTrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                evaluation_interval=int(training.get("evaluation_interval", 5)),
                patience=int(training["patience"]),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
            ),
            logger=logger,
        )
    elif model == "tricdr":
        trainer = TriCDRTrainer(
            cdsr_model,
            TriCDRTrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                lr_decay_rate=float(training.get("lr_decay_rate", 0.99)),
                patience=int(training["patience"]),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
            ),
            logger=logger,
        )
    elif model == "syncrec":
        trainer = SyNCRecTrainer(
            cdsr_model,
            SyNCRecTrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training.get("weight_decay", 0.0)),
                patience=int(training["patience"]),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
            ),
            logger=logger,
        )
    elif model == "autocdsr":
        from cdsr.trainers.autocdsr import AutoCDSRTrainer, AutoCDSRTrainerConfig

        trainer = AutoCDSRTrainer(
            cdsr_model,
            AutoCDSRTrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training.get("weight_decay", 1e-5)),
                max_steps=int(training.get("max_steps", 80000)),
                warmup_steps=int(training.get("warmup_steps", 1000)),
                min_lr_ratio=float(training.get("min_lr_ratio", 0.3)),
                precision=str(training.get("precision", "16-mixed")),
                patience=int(training["patience"]),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
            ),
            logger=logger,
        )
    elif model == "cmgr":
        trainer = CMGRTrainer(
            cdsr_model,
            CMGRTrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training.get("weight_decay", 0.0)),
                patience=int(training["patience"]),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
                detect_anomaly=bool(training.get("detect_anomaly", True)),
            ),
            logger=logger,
        )
    elif model == "hgtl":
        trainer = HGTLTrainer(
            cdsr_model,
            HGTLTrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                patience=int(training["patience"]),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
            ),
            logger=logger,
        )
    elif model == "horizonrec":
        trainer = HorizonRecTrainer(
            cdsr_model,
            HorizonRecTrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                patience=int(training["patience"]),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
            ),
            logger=logger,
        )
    elif model == "llm4cdsr":
        trainer = LLM4CDSRTrainer(
            cdsr_model,
            LLM4CDSRTrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training.get("weight_decay", 0.0)),
                patience=int(training["patience"]),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
            ),
            logger=logger,
        )
    elif model == "comm_cdsr_best":
        trainer = StandardTrainer(
            cdsr_model,
            TrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training.get("weight_decay", 0.0)),
                grad_clip_norm=training.get("grad_clip_norm"),
                device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
                patience=int(training["patience"]),
                lr_decay_step=int(training.get("lr_decay_step", 1)),
                lr_decay_rate=float(training.get("lr_decay_rate", 1.0)),
                swa_topk=int(training.get("swa_topk", 0)),
            ),
            logger=logger,
        )
    else:
        trainer = SemaCDRTrainer(
            cdsr_model,
            SemaCDRTrainerConfig(
                epochs=int(training["epochs"]),
                learning_rate=float(training["learning_rate"]),
                semantic_learning_rate=float(parameters.get("semantic_learning_rate", 3e-4)),
                weight_decay=float(training.get("weight_decay", 1e-4)),
                grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
                lr_decay_step=int(training.get("lr_decay_step", 5)),
                lr_decay_rate=float(training.get("lr_decay_rate", 0.98)),
                evaluation_interval=int(training.get("evaluation_interval", 5)),
                patience=int(training["patience"]), device=device,
                valid_metric=str(evaluation["valid_metric"]).lower(),
            ), logger=logger,
        )
    logger.info("Start %s training on %s", model.upper(), device)
    torch.cuda.synchronize(torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    training_started = time.perf_counter()
    rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if model == "amid":
        fit_result = trainer.fit(
            train_loader,
            dr_loader=dr_loader,
            valid_loader=valid_loader,
            evaluator=evaluator,
        )
    else:
        fit_result = trainer.fit(train_loader, valid_loader=valid_loader, evaluator=evaluator)
    torch.cuda.synchronize(torch.device(device))
    training_wall_seconds = time.perf_counter() - training_started
    rss_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    epoch_training_seconds = [
        float(record["training_seconds"])
        for record in fit_result.get("history", [])
        if "training_seconds" in record
    ]
    efficiency = {
        "training_wall_seconds": training_wall_seconds,
        "epoch_training_seconds": epoch_training_seconds,
        "epoch_training_seconds_sum": sum(epoch_training_seconds),
        "mean_epoch_training_seconds": (
            sum(epoch_training_seconds) / len(epoch_training_seconds)
            if epoch_training_seconds else None
        ),
        "process_peak_rss_mib": rss_after_kib / 1024.0,
        "process_peak_rss_increase_mib": max(0, rss_after_kib - rss_before_kib) / 1024.0,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(torch.device(device)) / 1024**2,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(torch.device(device)) / 1024**2,
        "community_preparation_seconds": round(community_prep_seconds, 3),
        "content_preparation_seconds": round(content_prep_seconds, 3),
    }
    logger.info(
        "Efficiency: training_wall_seconds=%.3f, epoch_training_seconds_sum=%.3f, "
        "process_peak_rss_mib=%.2f, cuda_peak_allocated_mib=%.2f, "
        "cuda_peak_reserved_mib=%.2f",
        efficiency["training_wall_seconds"],
        efficiency["epoch_training_seconds_sum"],
        efficiency["process_peak_rss_mib"],
        efficiency["cuda_peak_allocated_mib"],
        efficiency["cuda_peak_reserved_mib"],
    )
    validation_result = _best_validation_result(fit_result)
    test_result = None
    neural_test_before_rerank = None
    community_reranker = None
    if run_mode == "train_test":
        logger.info("Start testing on target domain %s", target_domain_name)
        evaluation_model = cdsr_model
        if model == "comm_cdsr_best" and bool(
            config.get("community_rerank", {}).get("enabled", False)
        ):
            from cdsr.models.community_rerank import CommunityReranker

            rerank_settings = dict(config["community_rerank"])
            # Rerank-only model override (see tune_eval site for rationale).
            rerank_api_settings = dict(config["community_memory"])
            if rerank_settings.get("chat_model"):
                rerank_api_settings["chat_model"] = rerank_settings["chat_model"]
            community_reranker = CommunityReranker(
                cdsr_model,
                corpus,
                entries,
                comm_features,
                rerank_settings,
                rerank_api_settings,
            )
            evaluation_model = community_reranker
            logger.info(
                "CommCDSR selected-checkpoint LLM Top-K: neural_top=%d, "
                "llm_top=%d, concurrency=%d, max_requests=%d",
                int(rerank_settings.get("candidate_size", 50)),
                int(rerank_settings.get("output_topk", 20)),
                int(rerank_settings.get("concurrency", 6)),
                int(rerank_settings.get("max_requests", 10000)),
            )
            logger.info("Evaluate CommCDSR neural ranking before LLM Top-K")
            neural_test_before_rerank = evaluator.evaluate(
                cdsr_model, test_loader, device=trainer.device, split="test"
            )
        test_result = evaluator.evaluate(
            evaluation_model, test_loader, device=trainer.device, split="test"
        )
        if community_reranker is not None:
            logger.info("CommCDSR LLM Top-K report: %s", community_reranker.report())
    else:
        logger.info(
            "TUNE MODE: test split is locked; best_epoch=%s, best_valid_score=%s",
            fit_result.get("best_epoch"), fit_result.get("best_valid_score"),
        )
        if (model == "comm_cdsr_best"
                and bool(config.get("community_rerank", {}).get("enabled", False))
                and bool(config.get("community_rerank", {}).get("tune_eval", False))):
            # Opt-in valid-side rerank measurement: same selected checkpoint,
            # valid split only. Used to tune rerank thresholds without
            # touching test.
            from cdsr.models.community_rerank import CommunityReranker

            rerank_settings = dict(config["community_rerank"])
            # Rerank-only model override: lets the decision LLM differ from
            # the profile-generation LLM without touching the feature
            # fingerprint (cached profiles stay valid, no regeneration).
            rerank_api_settings = dict(config["community_memory"])
            if rerank_settings.get("chat_model"):
                rerank_api_settings["chat_model"] = rerank_settings["chat_model"]
            community_reranker = CommunityReranker(
                cdsr_model,
                corpus,
                entries,
                comm_features,
                rerank_settings,
                rerank_api_settings,
            )
            neural_valid_at_best = evaluator.evaluate(
                cdsr_model, valid_loader, device=trainer.device, split="valid"
            )
            sweep = rerank_settings.get("tune_thresholds")
            topk_sweep = rerank_settings.get("tune_output_topks")
            gates_sweep = rerank_settings.get("tune_trigger_gates")
            prompt_versions = rerank_settings.get("tune_prompt_versions")
            if gates_sweep:
                # Trigger-gate diagnostic: evaluate the valid split under
                # several gate combinations (uncertainty / bridge / memory
                # thresholds). Meant to run with max_requests=0 so triggered
                # users fall back to neural order — no API cost, and the
                # stats/skip_reasons still show which gate blocks triggering.
                neural_valid_at_best = evaluator.evaluate(
                    cdsr_model, valid_loader, device=trainer.device, split="valid"
                )
                for gates in gates_sweep:
                    community_reranker.settings.update(gates)
                    community_reranker.stats = {key: 0 for key in community_reranker.stats}
                    community_reranker.skip_reasons = {}
                    community_reranker.uncertainty_samples = []
                    rerank_valid = evaluator.evaluate(
                        community_reranker, valid_loader, device=trainer.device, split="valid"
                    )
                    report = community_reranker.report()
                    logger.info(
                        "TUNE TRIGGER GATES %s | triggered=%s skipped=%s reasons=%s "
                        "uncertainty=%s | valid ndcg@10=%s hit@10=%s (neural %s hit %s)",
                        gates,
                        report["stats"].get("triggered_users"),
                        report["stats"].get("skipped"),
                        report.get("skip_reasons"),
                        report.get("uncertainty_percentiles"),
                        rerank_valid["overall"]["ndcg@10"],
                        rerank_valid["overall"].get("hit@10"),
                        neural_valid_at_best["overall"]["ndcg@10"],
                        neural_valid_at_best["overall"].get("hit@10"),
                    )
            elif topk_sweep:
                # output_topk sweep at the configured uncertainty threshold.
                # Each topk changes the request identity, so each tier bills
                # its own responses; training still happens only once.
                community_reranker.settings["uncertainty_threshold"] = float(
                    rerank_settings.get("uncertainty_threshold", 0.7)
                )
                measurements = []
                for topk in topk_sweep:
                    community_reranker.settings["output_topk"] = int(topk)
                    community_reranker.stats = {key: 0 for key in community_reranker.stats}
                    rerank_valid = evaluator.evaluate(
                        community_reranker, valid_loader, device=trainer.device, split="valid"
                    )
                    score = rerank_valid["overall"]["ndcg@10"]
                    measurements.append((int(topk), score))
                    logger.info(
                        "TUNE RERANK topk=%s neural valid ndcg@10=%s hit@10=%s | "
                        "rerank valid ndcg@10=%s hit@10=%s",
                        topk,
                        neural_valid_at_best["overall"]["ndcg@10"],
                        neural_valid_at_best["overall"].get("hit@10"),
                        score,
                        rerank_valid["overall"].get("hit@10"),
                    )
                    logger.info(
                        "CommCDSR tune-mode LLM Top-K report (topk=%s): %s",
                        topk, community_reranker.report(),
                    )
                best_topk, best_score = max(measurements, key=lambda pair: pair[1])
                logger.info(
                    "TUNE RERANK topk sweep %s | best topk=%s ndcg@10=%s",
                    [(value, round(score, 6)) for value, score in measurements],
                    best_topk, best_score,
                )
            elif prompt_versions:
                # Prompt A/B: same trained checkpoint, same candidates, same
                # valid users — the arms differ ONLY in the LLM-facing
                # prompt. Paired per-user deltas isolate the prompt effect
                # from user-level variance (non-triggered users cancel out).
                per_user_by_version: dict[int, dict[int, float]] = {}
                measurements = []
                for version in prompt_versions:
                    community_reranker.settings["prompt_version"] = int(version)
                    community_reranker.stats = {
                        key: 0 for key in community_reranker.stats
                    }
                    community_reranker.skip_reasons = {}
                    rerank_valid = evaluator.evaluate(
                        community_reranker, valid_loader,
                        device=trainer.device, split="valid",
                    )
                    report = community_reranker.report()
                    per_user_by_version[int(version)] = {
                        int(row["user_id"]): float(row["ndcg@10"])
                        for row in rerank_valid.get("per_user", [])
                    }
                    measurements.append({
                        "prompt_version": int(version),
                        "valid_ndcg@10": rerank_valid["overall"]["ndcg@10"],
                        "valid_hit@10": rerank_valid["overall"].get("hit@10"),
                        "triggered": report["stats"].get("triggered_users"),
                        "fallback": report["stats"].get("fallback"),
                    })
                    logger.info(
                        "TUNE PROMPT v%s | triggered=%s fallback=%s | valid ndcg@10=%s "
                        "(neural %s)",
                        version, report["stats"].get("triggered_users"),
                        report["stats"].get("fallback"),
                        rerank_valid["overall"]["ndcg@10"],
                        neural_valid_at_best["overall"]["ndcg@10"],
                    )
                if len(measurements) == 2:
                    version_a = int(measurements[0]["prompt_version"])
                    version_b = int(measurements[1]["prompt_version"])
                    map_a = per_user_by_version[version_a]
                    map_b = per_user_by_version[version_b]
                    deltas = [
                        map_b[user] - map_a[user]
                        for user in sorted(set(map_a) & set(map_b))
                    ]
                    changed = [delta for delta in deltas if abs(delta) > 1e-12]
                    mean = sum(deltas) / len(deltas) if deltas else 0.0
                    rng = random.Random(0)
                    boots = sorted(
                        sum(deltas[rng.randrange(len(deltas))] for _ in deltas)
                        / len(deltas)
                        for _ in range(2000)
                    ) if deltas else [0.0, 0.0]
                    logger.info(
                        "TUNE PROMPT A/B paired v%s->v%s | %d users (%d changed by "
                        "prompt) | mean delta ndcg@10=%+.6f | 95%% CI [%+.6f, "
                        "%+.6f] | wins=%d losses=%d",
                        version_a, version_b, len(deltas), len(changed), mean,
                        boots[int(0.025 * len(boots))],
                        boots[int(0.975 * len(boots))],
                        sum(delta > 0 for delta in changed),
                        sum(delta < 0 for delta in changed),
                    )
            elif sweep:
                rerank_valid = evaluator.evaluate(
                    community_reranker, valid_loader, device=trainer.device, split="valid"
                )
                logger.info(
                    "TUNE RERANK neural valid ndcg@10=%s | rerank valid ndcg@10=%s",
                    neural_valid_at_best["overall"]["ndcg@10"],
                    rerank_valid["overall"]["ndcg@10"],
                )
                logger.info("CommCDSR tune-mode LLM Top-K report: %s", community_reranker.report())

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = Path(config["paths"]["output_dir"]).resolve() / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = destination / "model.pt" if saved else None
    if checkpoint is not None:
        torch.save(cdsr_model.state_dict(), checkpoint)
        logger.info("Saving current: %s", checkpoint)
    quality_validation = None
    if (
        model == "comm_cdsr_best"
        and run_mode == "train_test"
        and bool(config.get("comm_validation", {}).get("enabled", False))
    ):
        from cdsr.validation.comm_cdsr_best import validate_comm_cdsr

        logger.info("Start CommCDSR three-module quality validation")
        validation_settings = dict(config["comm_validation"])
        validation_settings["community_memory"] = config["community_memory"]
        quality_validation = validate_comm_cdsr(
            model=cdsr_model,
            corpus=corpus,
            features=comm_features,
            domain_entries=entries,
            settings=validation_settings,
            evaluator=evaluator,
            test_loader=test_loader,
            device=trainer.device,
            output_dir=validation_settings["output_dir"],
            seed=protocol_seed,
            logger=logger,
            neural_test_result=neural_test_before_rerank,
            reranked_test_result=test_result if community_reranker is not None else None,
            rerank_report=(
                community_reranker.report() if community_reranker is not None else None
            ),
        )
    result = {
        "model": model,
        "run_mode": run_mode,
        "method_variant": (
            f"{model}_strict_target_only"
            if target_only else (
            f"{model}_{cdsr_model.memory_source}"
            if model == "comm_cdsr_best"
            else (
                "horizonrec_without_released_pretraining"
                if model == "horizonrec" and not bool(parameters.get("pretrain", False))
                else (
                    "horizonrec_regenerated_catalog_aligned_pretraining"
                    if model == "horizonrec"
                    else (
                        "cmgr_official_learned_embedding_branch"
                        if model == "cmgr" and not bool(parameters.get("local_emb", False))
                        and not bool(parameters.get("global_emb", False))
                        else (
                            "cmgr_catalog_aligned_llm_embeddings"
                            if model == "cmgr"
                            else (
                                "semacdr_appendix_a_reconstructed_semantics"
                                if model == "semacdr"
                                else "native_model_under_unified_protocol"
                            )
                        )
                    )
                )
            )
        )),
        "target_only": target_only,
        "smoke_test": smoke_enabled,
        "native_status": model_specs[model].native_status,
        "config_id": config.config_id,
        "run_seed": seed,
        "protocol_seed": protocol_seed,
        "data": str(Path(preparation["canonical_dir"]).resolve()),
        "protocol_id": protocol_bundle.protocol_id,
        "adapter_id": adapter_export.adapter_id,
        "target_domain": target_domain_name,
        "target_domain_id": target_domain_id,
        "user_group_counts": corpus.metadata.get("user_group_counts", {}),
        "training_examples": {
            "main": len(train_loader.dataset),
            "dr": len(dr_loader.dataset) if dr_loader is not None else None,
        },
        "side_features": (
            {
                "source": llm_features.source,
                "cache_dir": str(llm_features.cache_dir),
                "api_key_persisted": False,
            }
            if model == "llm4cdsr"
            else ({"source": semantic_features.source,
                   "cache_dir": str(semantic_features.cache_dir) if semantic_features.cache_dir else None,
                   "prompt_version": ((semantic_features.manifest or {}).get("prompt_version")),
                   "paper_result_eligible": (
                       semantic_features.source == "aligned_files"
                       or (semantic_features.source in {"api", "api_cache", "frozen_api_cache"}
                           and bool(config.get("semantic_features", {}).get(
                               "acknowledge_reconstructed_schema", False)))
                   ),
                   "reproduction_note": ((semantic_features.manifest or {}).get("paper_reproduction_note")),
                   "api_key_persisted": False}
                  if model == "semacdr" else (
                      {
                          "source": comm_features.source,
                          "cache_dir": str(comm_features.cache_dir),
                          "profile_version": comm_features.manifest.get("profile_version"),
                          "schema_version": comm_features.manifest.get("schema_version"),
                          "source_split": comm_features.manifest.get("source_split"),
                          "profile_signal": comm_features.manifest.get("profile_signal"),
                          "formation": comm_features.manifest.get("formation"),
                          "community_sizes_by_domain": comm_features.manifest.get(
                              "community_sizes_by_domain"
                          ),
                          "graph_profile_separation": comm_features.manifest.get(
                              "graph_profile_separation"
                          ),
                          "api_key_persisted": False,
                      }
                      if model == "comm_cdsr_best" else None
                  ))
        ),
        "fit": fit_result,
        "validation": validation_result,
        "efficiency": efficiency,
        "llm_token_usage": (
            {
                "this_run": dict(llm_features.api_usage),
                "cache_cumulative": dict(llm_features.api_usage_cumulative),
            }
            if model == "llm4cdsr"
            else (
                {
                    "this_run": dict(semantic_features.api_usage),
                    "cache_cumulative": dict(semantic_features.api_usage_cumulative),
                }
                if model == "semacdr" else (
                    {
                        "this_run": dict(comm_features.api_usage),
                        "cache_cumulative": dict(comm_features.api_usage_cumulative),
                        "reranking_this_run": (
                            community_reranker.report()["token_usage_this_run"]
                            if community_reranker is not None else {}
                        ),
                        "reranking_cache_cumulative": (
                            community_reranker.report()["token_usage_cumulative"]
                            if community_reranker is not None else {}
                        ),
                    }
                    if model == "comm_cdsr_best" else {}
                )
            )
        ),
        "test": test_result,
        "neural_test_before_rerank": neural_test_before_rerank,
        "community_rerank": (
            community_reranker.report() if community_reranker is not None else {
                "enabled": False
            }
        ),
        "quality_validation": quality_validation,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "log_file": str(log_file),
    }
    result_file = destination / "result.json"
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if test_result is not None:
        logger.info("test result (overall):\n%s", _format_metrics(test_result["overall"]))
        for group in ("overlap", "non_overlap"):
            metrics = test_result[group]
            logger.info(
                "test result (%s, count=%d):\n%s",
                group,
                int(metrics.get("count", 0)),
                _format_metrics(metrics) or "<empty>",
            )
    logger.info("Result file: %s", result_file)
    return result
