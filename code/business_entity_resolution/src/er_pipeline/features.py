"""Fast vectorized pair features for candidate ownership scoring."""

from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from .text import LEGAL_WORDS


FEATURE_COLUMNS = [
    "tfidf_score", "word_tfidf_score", "char_tfidf_score",
    "retrieval_rank", "score_from_best", "score_ratio",
    "name_jaccard", "name_containment", "address_jaccard", "address_containment",
    "number_jaccard", "number_containment", "name_exact", "address_exact",
    "name_core_exact", "name_core_jaccard", "postal_exact", "postal_conflict",
    "house_number_exact", "house_number_conflict", "all_numbers_exact",
    "name_length_ratio", "address_length_ratio", "name_edit", "name_weighted",
    "name_token_set", "address_weighted", "address_token_set", "combined_weighted",
    "target_address_missing", "reference_address_missing", "source_is_3",
]


def _ratio(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    maximum = pl.max_horizontal(left, right)
    minimum = pl.min_horizontal(left, right)
    return pl.when(maximum > 0).then(minimum / maximum).otherwise(0.0)


def _overlap(prefix: str) -> list[pl.Expr]:
    left, right = pl.col(f"l_{prefix}_tokens"), pl.col(f"r_{prefix}_tokens")
    intersection = left.list.set_intersection(right).list.len().cast(pl.Float32)
    union = left.list.set_union(right).list.len().cast(pl.Float32)
    smaller = pl.min_horizontal(left.list.len(), right.list.len()).cast(pl.Float32)
    return [
        pl.when(union > 0).then(intersection / union).otherwise(0.0).alias(f"{prefix}_jaccard"),
        pl.when(smaller > 0).then(intersection / smaller).otherwise(0.0).alias(f"{prefix}_containment"),
    ]


def build_pair_features(candidates: pl.DataFrame, targets: pl.DataFrame,
                        reference: pl.DataFrame, target_source: int) -> pl.DataFrame:
    legal_pattern = r"\b(?:" + "|".join(LEGAL_WORDS) + r")\b"
    pairs = (
        candidates.join(
            targets.select(
                "right_idx", pl.col("entity_id").alias("target_entity_id"),
                pl.col("name_norm").alias("r_name"),
                pl.col("address_norm").alias("r_address"),
            ),
            on="right_idx", how="left",
        )
        .join(
            reference.select(
                "left_idx", pl.col("entity_id").alias("source1_entity_id"),
                pl.col("name_norm").alias("l_name"),
                pl.col("address_norm").alias("l_address"),
            ),
            on="left_idx", how="left",
        )
        .with_columns(
            pl.col("l_name").str.split(" ").list.eval(
                pl.element().filter(pl.element() != "")
            ).list.unique().alias("l_name_tokens"),
            pl.col("r_name").str.split(" ").list.eval(
                pl.element().filter(pl.element() != "")
            ).list.unique().alias("r_name_tokens"),
            pl.col("l_address").str.split(" ").list.eval(
                pl.element().filter(pl.element() != "")
            ).list.unique().alias("l_address_tokens"),
            pl.col("r_address").str.split(" ").list.eval(
                pl.element().filter(pl.element() != "")
            ).list.unique().alias("r_address_tokens"),
            pl.col("l_address").str.extract_all(r"\d+").list.unique().alias("l_number_tokens"),
            pl.col("r_address").str.extract_all(r"\d+").list.unique().alias("r_number_tokens"),
            pl.col("l_address").str.extract(r"\b(\d{5,6})\b").fill_null("").alias("l_postal"),
            pl.col("r_address").str.extract(r"\b(\d{5,6})\b").fill_null("").alias("r_postal"),
            pl.col("l_address").str.extract(r"\b(\d+)\b").fill_null("").alias("l_house"),
            pl.col("r_address").str.extract(r"\b(\d+)\b").fill_null("").alias("r_house"),
            pl.col("l_name").str.replace_all(legal_pattern, " ").str.replace_all(
                r"\s+", " "
            ).str.strip_chars().alias("l_name_core"),
            pl.col("r_name").str.replace_all(legal_pattern, " ").str.replace_all(
                r"\s+", " "
            ).str.strip_chars().alias("r_name_core"),
        )
    )
    pairs = pairs.with_columns(
        pl.col("l_name_core").str.split(" ").list.eval(
            pl.element().filter(pl.element() != "")
        ).list.unique().alias("l_name_core_tokens"),
        pl.col("r_name_core").str.split(" ").list.eval(
            pl.element().filter(pl.element() != "")
        ).list.unique().alias("r_name_core_tokens"),
    )
    best = pl.col("tfidf_score").max().over("right_idx")
    pairs = pairs.with_columns(
        pl.lit(target_source, dtype=pl.UInt8).alias("target_source"),
        (pl.lit(target_source * 10_000_000, dtype=pl.UInt64)
         + pl.col("right_idx").cast(pl.UInt64)).alias("target_key"),
        (best - pl.col("tfidf_score")).alias("score_from_best"),
        (pl.col("tfidf_score") / best.clip(lower_bound=1e-6)).alias("score_ratio"),
        *_overlap("name"), *_overlap("address"), *_overlap("number"),
        *_overlap("name_core"),
        ((pl.col("l_name") == pl.col("r_name")) & (pl.col("l_name") != "")).cast(pl.Float32).alias("name_exact"),
        ((pl.col("l_address") == pl.col("r_address")) & (pl.col("l_address") != "")).cast(pl.Float32).alias("address_exact"),
        ((pl.col("l_name_core") == pl.col("r_name_core")) & (pl.col("l_name_core") != "")).cast(pl.Float32).alias("name_core_exact"),
        ((pl.col("l_postal") == pl.col("r_postal")) & (pl.col("l_postal") != "")).cast(pl.Float32).alias("postal_exact"),
        ((pl.col("l_postal") != pl.col("r_postal")) & (pl.col("l_postal") != "") & (pl.col("r_postal") != "")).cast(pl.Float32).alias("postal_conflict"),
        ((pl.col("l_house") == pl.col("r_house")) & (pl.col("l_house") != "")).cast(pl.Float32).alias("house_number_exact"),
        ((pl.col("l_house") != pl.col("r_house")) & (pl.col("l_house") != "") & (pl.col("r_house") != "")).cast(pl.Float32).alias("house_number_conflict"),
        ((pl.col("l_number_tokens") == pl.col("r_number_tokens")) &
         (pl.col("l_number_tokens").list.len() > 0)).cast(pl.Float32).alias("all_numbers_exact"),
        _ratio(pl.col("l_name").str.len_chars(), pl.col("r_name").str.len_chars()).alias("name_length_ratio"),
        _ratio(pl.col("l_address").str.len_chars(), pl.col("r_address").str.len_chars()).alias("address_length_ratio"),
        (pl.col("r_address") == "").cast(pl.Float32).alias("target_address_missing"),
        (pl.col("l_address") == "").cast(pl.Float32).alias("reference_address_missing"),
        pl.lit(float(target_source == 3), dtype=pl.Float32).alias("source_is_3"),
    )
    left_names, right_names = pairs["l_name"].to_list(), pairs["r_name"].to_list()
    left_addresses = pairs["l_address"].to_list()
    right_addresses = pairs["r_address"].to_list()
    pairs = pairs.with_columns(
        pl.Series("name_edit", np.fromiter(
            (fuzz.ratio(a, b) / 100.0 for a, b in zip(left_names, right_names)),
            dtype=np.float32, count=len(pairs),
        )),
        pl.Series("name_weighted", np.fromiter(
            (fuzz.WRatio(a, b) / 100.0 for a, b in zip(left_names, right_names)),
            dtype=np.float32, count=len(pairs),
        )),
        pl.Series("name_token_set", np.fromiter(
            (fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(left_names, right_names)),
            dtype=np.float32, count=len(pairs),
        )),
        pl.Series("address_weighted", np.fromiter(
            (fuzz.WRatio(a, b) / 100.0 for a, b in zip(left_addresses, right_addresses)),
            dtype=np.float32, count=len(pairs),
        )),
        pl.Series("address_token_set", np.fromiter(
            (fuzz.token_set_ratio(a, b) / 100.0
             for a, b in zip(left_addresses, right_addresses)),
            dtype=np.float32, count=len(pairs),
        )),
        pl.Series("combined_weighted", np.fromiter(
            (fuzz.WRatio(f"{a} {c}", f"{b} {d}") / 100.0
             for a, b, c, d in zip(
                 left_names, right_names, left_addresses, right_addresses
             )),
            dtype=np.float32, count=len(pairs),
        )),
    )
    return pairs.with_columns(
        pl.col("retrieval_rank").cast(pl.Float32),
        *[pl.col(column).fill_nan(0).fill_null(0).cast(pl.Float32) for column in FEATURE_COLUMNS
          if column not in {"retrieval_rank"}],
    )
