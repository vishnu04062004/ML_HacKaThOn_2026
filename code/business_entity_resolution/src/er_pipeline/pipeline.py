"""End-to-end training, inference, output assembly, and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import shutil
import time
import warnings

import joblib
import numpy as np
import polars as pl

from .features import FEATURE_COLUMNS, build_pair_features
from .modeling import make_model, target_decisions, tune_threshold
from .retrieval import fit_country_index, retrieve_country_chunks
from .text import add_combined_word_text


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


RECORD_SCHEMA = {
    "entity_id": pl.String, "business_name": pl.String,
    "business_address": pl.String, "country": pl.String,
}


@dataclass
class Config:
    data_dir: Path
    output_dir: Path
    training_targets_per_source: int = 250_000
    retrieval_top_n: int = 8
    chunk_size: int = 10_000
    threads: int = 8
    seed: int = 2026
    resume: bool = False
    train_only: bool = False
    resume_threshold: float | None = None
    resume_margin: float | None = None


def read_records(path: Path, index_name: str) -> pl.DataFrame:
    return (
        pl.read_csv(path, separator="\t", schema_overrides=RECORD_SCHEMA, null_values=[])
        .with_columns(
            pl.col("business_name").fill_null(""),
            pl.col("business_address").fill_null(""),
            pl.col("country").fill_null(""),
        )
        .with_row_index(index_name)
    )


def read_country_records(path: Path, index_name: str, country: str) -> pl.DataFrame:
    return (
        pl.scan_csv(path, separator="\t", schema_overrides=RECORD_SCHEMA, null_values=[])
        .with_row_index(index_name)
        .filter(pl.col("country") == country)
        .with_columns(
            pl.col("business_name").fill_null(""),
            pl.col("business_address").fill_null(""),
            pl.col("country").fill_null(""),
        )
        .collect(engine="streaming")
    )


def sample_records(path: Path, index_name: str, wanted: int, offset: int) -> pl.DataFrame:
    scan = pl.scan_csv(path, separator="\t", schema_overrides=RECORD_SCHEMA, null_values=[]).with_row_index(index_name)
    total = scan.select(pl.len()).collect(engine="streaming").item()
    step = max(1, total // wanted)
    return (
        scan.gather_every(step, offset=offset % step).head(wanted)
        .with_columns(
            pl.col("business_name").fill_null(""),
            pl.col("business_address").fill_null(""),
            pl.col("country").fill_null(""),
        )
        .collect(engine="streaming")
    )


def truth_for_targets(ground_truth: Path, targets: pl.DataFrame, source: int) -> pl.DataFrame:
    links = (
        pl.scan_csv(
            ground_truth, separator="\t",
            schema_overrides={"source1_entity_id": pl.String, "matched_entity_ids": pl.String},
            null_values=[],
        )
        .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids").str.starts_with(f"S{source}-"))
        .select(
            pl.col("matched_entity_ids").alias("target_entity_id"),
            pl.col("source1_entity_id").alias("true_owner"),
        )
    )
    return (
        targets.select("right_idx", pl.col("entity_id").alias("target_entity_id"))
        .lazy().join(links, on="target_entity_id", how="left")
        .with_columns(
            pl.col("true_owner").fill_null(""),
            (pl.lit(source * 10_000_000, dtype=pl.UInt64)
             + pl.col("right_idx").cast(pl.UInt64)).alias("target_key"),
        )
        .collect(engine="streaming")
    )


def _training_pairs(reference: pl.DataFrame, samples: dict[int, pl.DataFrame],
                    truths: dict[int, pl.DataFrame], config: Config) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    countries = sorted(reference["country"].unique().to_list())
    for country in countries:
        ref_country, index = fit_country_index(reference, country)
        reference_rows = ref_country["left_idx"].to_numpy()
        print(
            f"  training index {country}: word={index.word_matrix.shape}/"
            f"{index.word_matrix.nnz:,} nnz, char={index.char_matrix.shape}/"
            f"{index.char_matrix.nnz:,} nnz", flush=True,
        )
        for source in (2, 3):
            for chunk, candidates in retrieve_country_chunks(
                samples[source], country, reference_rows, index,
                config.retrieval_top_n, config.chunk_size, config.threads,
            ):
                if candidates.is_empty():
                    processed += chunk.height
                    completed[task_key] = processed
                    resume_state.update({
                        "candidate_part": candidate_part,
                        "match_part": match_part,
                        "completed": completed,
                        "legacy_completed_chunks": 0,
                        "threshold": threshold,
                        "probability_margin": probability_margin,
                    })
                    _save_resume_state(resume_path, resume_state)
                    print(f"    {country} S{source}: {processed:,} targets", flush=True)
                    continue
                features = build_pair_features(candidates, chunk, reference, source)
                features = (
                    features.join(
                        truths[source].select("target_key", "true_owner"),
                        on="target_key", how="left",
                    )
                    .with_columns(
                        (pl.col("source1_entity_id") == pl.col("true_owner"))
                        .cast(pl.UInt8).alias("label")
                    )
                    .select(
                        "target_key", "target_entity_id", "source1_entity_id",
                        "true_owner", "label", *FEATURE_COLUMNS,
                    )
                )
                parts.append(features)
    return pl.concat(parts, how="vertical_relaxed").rechunk()


def train_model(train_dir: Path, config: Config):
    print("[1/5] Loading reference and sampled training targets", flush=True)
    reference = add_combined_word_text(read_records(train_dir / "train_source1.tsv", "left_idx"))
    samples, truths = {}, {}
    for source in (2, 3):
        samples[source] = add_combined_word_text(sample_records(
            train_dir / f"train_source{source}.tsv", "right_idx",
            config.training_targets_per_source, offset=source - 2,
        ))
        truths[source] = truth_for_targets(
            train_dir / "train_ground_truth.tsv", samples[source], source
        )

    print("[2/5] Retrieving and featurizing sampled training pairs", flush=True)
    pairs = _training_pairs(reference, samples, truths, config)
    owner_counts = (
        pairs.select("target_key", "true_owner").unique()
        .filter(pl.col("true_owner") != "")
        .group_by("true_owner")
        .agg(pl.len().alias("owner_target_count"))
    )
    pairs = pairs.join(owner_counts, on="true_owner", how="left").with_columns(
        pl.when(pl.col("true_owner") != "")
        .then(1.0 / pl.col("owner_target_count").fill_null(1).sqrt())
        .otherwise(1.0)
        .cast(pl.Float32)
        .alias("entity_weight")
    )
    actual_positives = sum(t.filter(pl.col("true_owner") != "").height for t in truths.values())
    retrieved_positives = pairs.filter(pl.col("label") == 1)["target_key"].n_unique()
    blocking_recall = retrieved_positives / max(1, actual_positives)
    # Keep all sampled links belonging to the same S1 owner in the same fold;
    # the leaderboard metric also treats each S1 match set as one unit.
    validation_hash = pl.when(pl.col("true_owner") != "").then(
        pl.col("true_owner").hash(seed=config.seed)
    ).otherwise(pl.col("target_key").hash(seed=config.seed))
    valid_mask = pairs.select(
        ((validation_hash % 5) == 0).alias("is_valid")
    )["is_valid"]
    train = pairs.filter(~valid_mask)
    valid = pairs.filter(valid_mask)
    print(
        f"  sampled pairs={pairs.height:,}, positives={pairs['label'].sum():,}, "
        f"fused candidate recall={blocking_recall:.5f}", flush=True,
    )

    model = make_model(config.seed)
    y_train = train["label"].to_numpy()
    positive_weight = min(4.0, max(1.0, np.sqrt((y_train == 0).sum() / max(1, y_train.sum()))))
    weights = (
        np.where(y_train == 1, positive_weight, 1.0)
        * train["entity_weight"].to_numpy()
    )
    model.fit(train.select(FEATURE_COLUMNS).to_numpy(), y_train, sample_weight=weights)
    valid = valid.with_columns(
        pl.Series("probability", model.predict_proba(valid.select(FEATURE_COLUMNS).to_numpy())[:, 1])
    )
    valid_truth = pl.concat([
        truth.select("target_key", "true_owner").filter(
            (pl.when(pl.col("true_owner") != "").then(
                pl.col("true_owner").hash(seed=config.seed)
            ).otherwise(pl.col("target_key").hash(seed=config.seed)) % 5) == 0
        )
        for truth in truths.values()
    ])
    threshold, probability_margin, validation = tune_threshold(valid, valid_truth)
    print(
        f"  validation macro F0.5={validation['f0_5']:.5f}, "
        f"precision={validation['precision']:.5f}, recall={validation['recall']:.5f}, "
        f"threshold={threshold:.5f}, margin={probability_margin:.5f}", flush=True,
    )

    final_model = make_model(config.seed)
    y_all = pairs["label"].to_numpy()
    all_weights = (
        np.where(y_all == 1, positive_weight, 1.0)
        * pairs["entity_weight"].to_numpy()
    )
    final_model.fit(pairs.select(FEATURE_COLUMNS).to_numpy(), y_all, sample_weight=all_weights)
    metrics = {
        "sampled_training_pairs": pairs.height,
        "sampled_positive_targets": actual_positives,
        "blocking_recall_top_n": blocking_recall,
        "validation_macro_f0_5": validation["f0_5"],
        "validation_precision": validation["precision"],
        "validation_recall": validation["recall"],
        "probability_threshold": threshold,
        "probability_margin": probability_margin,
    }
    del pairs, train, valid, samples, truths
    gc.collect()
    return reference, final_model, threshold, probability_margin, metrics


def _write_part(frame: pl.DataFrame, directory: Path, prefix: str, number: int) -> None:
    if not frame.is_empty():
        frame.write_parquet(directory / f"{prefix}_{number:05d}.parquet", compression="zstd")


def _save_resume_state(path: Path, state: dict) -> None:
    """Atomically persist progress after a completed inference chunk."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_inference(reference: pl.DataFrame, model, threshold: float,
                  probability_margin: float,
                  test_dir: Path, cache_dir: Path, config: Config,
                  resume_path: Path, resume_state: dict) -> dict:
    candidate_part = int(resume_state.get("candidate_part", 0))
    match_part = int(resume_state.get("match_part", candidate_part))
    completed: dict[str, int] = {
        key: int(value) for key, value in resume_state.get("completed", {}).items()
    }
    # Old interrupted runs predate resume_state.json. Their sequential candidate
    # part count still identifies the exact completed chunk boundary.
    legacy_chunks_remaining = int(resume_state.get("legacy_completed_chunks", 0))
    countries = sorted(reference["country"].unique().to_list())
    for country in countries:
        ref_country, index = fit_country_index(reference, country)
        reference_rows = ref_country["left_idx"].to_numpy()
        print(
            f"  inference index {country}: word={index.word_matrix.shape}/"
            f"{index.word_matrix.nnz:,} nnz, char={index.char_matrix.shape}/"
            f"{index.char_matrix.nnz:,} nnz", flush=True,
        )
        for source in (2, 3):
            country_targets = add_combined_word_text(read_country_records(
                test_dir / f"test_source{source}.tsv", "right_idx", country
            ))
            task_key = f"{country}:S{source}"
            if task_key in completed:
                start_offset = min(completed[task_key], country_targets.height)
            elif legacy_chunks_remaining > 0:
                total_chunks = (
                    country_targets.height + config.chunk_size - 1
                ) // config.chunk_size
                completed_chunks = min(legacy_chunks_remaining, total_chunks)
                legacy_chunks_remaining -= completed_chunks
                start_offset = min(
                    completed_chunks * config.chunk_size, country_targets.height
                )
                completed[task_key] = start_offset
            else:
                start_offset = 0
            processed = start_offset
            if start_offset:
                print(
                    f"    resuming {country} S{source} at {start_offset:,}/"
                    f"{country_targets.height:,}", flush=True,
                )
            for chunk, candidates in retrieve_country_chunks(
                country_targets, country, reference_rows, index,
                config.retrieval_top_n, config.chunk_size, config.threads,
                start_offset=start_offset,
            ):
                if candidates.is_empty():
                    continue
                features = build_pair_features(candidates, chunk, reference, source)
                _write_part(
                    features.select("left_idx", "target_entity_id").unique(),
                    cache_dir, "candidate", candidate_part,
                )
                candidate_part += 1
                features = features.with_columns(
                    pl.Series("probability", model.predict_proba(
                        features.select(FEATURE_COLUMNS).to_numpy()
                    )[:, 1])
                )
                decisions = target_decisions(features, threshold, probability_margin)
                _write_part(
                    decisions.select("left_idx", "target_entity_id"),
                    cache_dir, "match", match_part,
                )
                match_part += 1
                processed += chunk.height
                completed[task_key] = processed
                resume_state.update({
                    "candidate_part": candidate_part,
                    "match_part": match_part,
                    "completed": completed,
                    "legacy_completed_chunks": 0,
                    "threshold": threshold,
                    "probability_margin": probability_margin,
                })
                _save_resume_state(resume_path, resume_state)
                print(f"    {country} S{source}: {processed:,} targets", flush=True)
            del country_targets
            gc.collect()
        del ref_country, index
        gc.collect()
    counts = {"candidate_parts": candidate_part, "match_parts": match_part}
    return counts


def assemble_output(reference: pl.DataFrame, cache_dir: Path, output_dir: Path) -> None:
    def assemble(pattern: str, value_column: str) -> pl.DataFrame:
        files = list(cache_dir.glob(pattern))
        if not files:
            grouped = pl.DataFrame({"left_idx": [], value_column: []},
                                   schema={"left_idx": pl.UInt32, value_column: pl.String})
        else:
            grouped = (
                pl.scan_parquet(files)
                .unique(["left_idx", "target_entity_id"])
                .group_by("left_idx")
                .agg(pl.col("target_entity_id").sort().alias("ids"))
                .with_columns(pl.col("ids").list.join(",").alias(value_column))
                .select("left_idx", value_column)
                .collect(engine="streaming")
            )
        return (
            reference.select("left_idx", pl.col("entity_id").alias("source1_entity_id"))
            .join(grouped, on="left_idx", how="left")
            .with_columns(pl.col(value_column).fill_null(""))
            .select("source1_entity_id", value_column)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    assemble("match_*.parquet", "matched_entity_ids").write_csv(
        output_dir / "matching_results.tsv", separator="\t", quote_style="never"
    )
    assemble("candidate_*.parquet", "candidate_entity_ids").write_csv(
        output_dir / "candidate_pairs.tsv", separator="\t", quote_style="never"
    )


def run(config: Config) -> dict:
    started = time.time()
    train_dir, test_dir = config.data_dir / "train", config.data_dir / "test"
    required = [train_dir / "train_source1.tsv", train_dir / "train_ground_truth.tsv",
                test_dir / "test_source1.tsv"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"--data-dir must be the official dataset directory: {config.data_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.output_dir / ".cache"
    resume_path = config.output_dir / "resume_state.json"
    model_path = config.output_dir / "ownership_model.joblib"
    if cache_dir.exists() and not config.resume:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = None
    resume_state: dict = {}
    if config.resume and resume_path.is_file() and model_path.is_file():
        resume_state = json.loads(resume_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
        threshold = float(resume_state["threshold"])
        probability_margin = float(resume_state.get("probability_margin", 0.0))
        metrics = resume_state.get("metrics", {})
        print(
            f"[resume] Loaded checkpoint with {resume_state.get('candidate_part', 0):,} "
            "completed chunks", flush=True,
        )
    elif config.resume and model_path.is_file() and any(cache_dir.glob("candidate_*.parquet")):
        candidate_numbers = [
            int(path.stem.rsplit("_", 1)[1])
            for path in cache_dir.glob("candidate_*.parquet")
        ]
        next_part = max(candidate_numbers) + 1
        if config.resume_threshold is not None:
            model = joblib.load(model_path)
            threshold = float(config.resume_threshold)
            probability_margin = float(config.resume_margin or 0.0)
            metrics = {"legacy_resume_used_supplied_threshold": True}
        else:
            print(
                "[resume] Legacy cache found. Reproducing training calibration once; "
                "completed inference chunks will not be rerun.", flush=True,
            )
            reference, model, threshold, probability_margin, metrics = train_model(
                train_dir, config
            )
            joblib.dump(model, model_path)
            del reference
            gc.collect()
        resume_state = {
            "candidate_part": next_part,
            "match_part": next_part,
            "completed": {},
            "legacy_completed_chunks": next_part,
            "threshold": threshold,
            "probability_margin": probability_margin,
            "metrics": metrics,
        }
        _save_resume_state(resume_path, resume_state)
        print(
            f"[resume] Recovered {next_part:,} cached chunks", flush=True,
        )
    else:
        reference, model, threshold, probability_margin, metrics = train_model(
            train_dir, config
        )
        joblib.dump(model, model_path)
        del reference
        gc.collect()
        resume_state = {
            "candidate_part": 0,
            "match_part": 0,
            "completed": {},
            "legacy_completed_chunks": 0,
            "threshold": threshold,
            "probability_margin": probability_margin,
            "metrics": metrics,
        }
        _save_resume_state(resume_path, resume_state)

    if config.train_only:
        metrics["elapsed_seconds"] = round(time.time() - started, 2)
        metrics["config"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        }
        (config.output_dir / "run_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(
            "[train-only] Model and calibrated macro-F0.5 checkpoint saved; "
            "run again with --resume for inference.", flush=True,
        )
        return metrics

    print("[3/5] Running full reverse retrieval and ownership inference", flush=True)
    test_reference = add_combined_word_text(read_records(test_dir / "test_source1.tsv", "left_idx"))
    metrics.update(run_inference(
        test_reference, model, threshold, probability_margin,
        test_dir, cache_dir, config, resume_path, resume_state,
    ))
    print("[4/5] Assembling required TSV files", flush=True)
    assemble_output(test_reference, cache_dir, config.output_dir)
    metrics["elapsed_seconds"] = round(time.time() - started, 2)
    metrics["config"] = {key: str(value) if isinstance(value, Path) else value
                         for key, value in asdict(config).items()}
    (config.output_dir / "run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    shutil.rmtree(cache_dir)
    resume_path.unlink(missing_ok=True)
    print("[5/5] Pipeline complete", flush=True)
    return metrics
