"""Reverse (target-to-reference) sparse top-N candidate retrieval."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import gc
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


@dataclass
class CountryIndex:
    word_vectorizer: TfidfVectorizer
    word_matrix: csr_matrix
    char_vectorizer: TfidfVectorizer
    char_matrix: csr_matrix


def fit_country_index(reference: pl.DataFrame, country: str) -> tuple[pl.DataFrame, CountryIndex]:
    subset = reference.filter(pl.col("country") == country)
    word_vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        sublinear_tf=True,
        norm="l2",
        max_df=0.03,
        dtype=np.float32,
    )
    word_matrix = word_vectorizer.fit_transform(subset["retrieval_text"].to_list()).tocsr()
    char_vectorizer = TfidfVectorizer(
        lowercase=False, analyzer="char_wb", ngram_range=(3, 4),
        min_df=2, max_df=0.25, max_features=300_000,
        sublinear_tf=True, norm="l2", dtype=np.float32,
    )
    char_matrix = char_vectorizer.fit_transform(subset["name_norm"].to_list()).tocsr()
    return subset, CountryIndex(
        word_vectorizer, word_matrix, char_vectorizer, char_matrix
    )


def _csr_candidates(similarities: csr_matrix, target_rows: np.ndarray,
                    reference_rows: np.ndarray, score_column: str) -> pl.DataFrame:
    counts = np.diff(similarities.indptr)
    if similarities.nnz == 0:
        return pl.DataFrame(
            schema={"right_idx": pl.UInt32, "left_idx": pl.UInt32,
                    score_column: pl.Float32}
        )
    return pl.DataFrame({
        "right_idx": np.repeat(target_rows, counts).astype(np.uint32),
        "left_idx": reference_rows[similarities.indices].astype(np.uint32),
        score_column: similarities.data.astype(np.float32),
    })


def retrieve_country_chunks(
    targets: pl.DataFrame,
    country: str,
    reference_rows: np.ndarray,
    index: CountryIndex,
    top_n: int,
    chunk_size: int,
    threads: int,
    start_offset: int = 0,
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
    country_targets = targets.filter(pl.col("country") == country)
    for start in range(start_offset, country_targets.height, chunk_size):
        chunk = country_targets.slice(start, chunk_size)
        word_query = index.word_vectorizer.transform(chunk["retrieval_text"].to_list()).tocsr()
        word_similarities = sp_matmul_topn(
            word_query,
            index.word_matrix.T,
            top_n=top_n,
            threshold=0.001,
            sort=True,
            n_threads=threads,
        )
        char_top_n = max(2, top_n // 2)
        char_query = index.char_vectorizer.transform(chunk["name_norm"].to_list()).tocsr()
        char_similarities = sp_matmul_topn(
            char_query, index.char_matrix.T, top_n=char_top_n,
            threshold=0.03, sort=True, n_threads=threads,
        )
        target_rows = chunk["right_idx"].to_numpy()
        word_candidates = _csr_candidates(
            word_similarities, target_rows, reference_rows, "word_tfidf_score"
        ).with_columns(pl.lit(0.0, dtype=pl.Float32).alias("char_tfidf_score"))
        char_candidates = _csr_candidates(
            char_similarities, target_rows, reference_rows, "char_tfidf_score"
        ).with_columns(pl.lit(0.0, dtype=pl.Float32).alias("word_tfidf_score"))
        candidates = (
            pl.concat([word_candidates, char_candidates], how="diagonal_relaxed")
            .group_by("right_idx", "left_idx")
            .agg(
                pl.col("word_tfidf_score").max(),
                pl.col("char_tfidf_score").max(),
            )
            .with_columns(
                pl.max_horizontal(
                    pl.col("word_tfidf_score"),
                    pl.col("char_tfidf_score") * 0.92,
                ).alias("tfidf_score")
            )
            .with_columns(
                (pl.col("tfidf_score").rank("ordinal", descending=True)
                 .over("right_idx") - 1).cast(pl.UInt8).alias("retrieval_rank")
            )
        )
        yield chunk, candidates
        del word_query, char_query, word_similarities, char_similarities
        del word_candidates, char_candidates, candidates
        gc.collect()
