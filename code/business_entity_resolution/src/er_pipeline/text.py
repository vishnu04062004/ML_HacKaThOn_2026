"""Vectorized, open-country text normalization expressions."""

from __future__ import annotations

import polars as pl


LEGAL_WORDS = [
    "the", "and", "co", "company", "corporation", "corp", "incorporated", "inc",
    "limited", "ltd", "private", "pvt", "llc", "llp", "plc", "services", "service",
    "group", "enterprises", "enterprise", "sa", "sas", "sarl", "eurl",
]
ADDRESS_WORDS = [
    "the", "and", "road", "rd", "street", "st", "avenue", "ave", "boulevard",
    "blvd", "near", "nr", "floor", "flr", "building", "bldg", "unit", "apartment",
    "apt", "drive", "dr", "lane", "ln", "highway", "hwy", "nagar", "marg",
    "rue", "route", "chemin", "cedex",
]


def normalized(column: str) -> pl.Expr:
    """Return conservative normalization while retaining all Unicode letters."""
    expr = pl.col(column).fill_null("").str.to_lowercase().str.replace_all("&", " and ")
    # Explicit Latin folding helps unseen French without damaging Indic scripts.
    replacements = [
        (r"[àáâäãå]", "a"), (r"[ç]", "c"), (r"[èéêë]", "e"),
        (r"[ìíîï]", "i"), (r"[ñ]", "n"), (r"[òóôöõ]", "o"),
        (r"[ùúûü]", "u"), (r"[ýÿ]", "y"), (r"[œ]", "oe"), (r"[æ]", "ae"),
    ]
    # Explicit Unicode literals repair the legacy mojibake table above and
    # improve matching for accented French and other Latin-script names.
    replacements.extend([
        (r"[àáâäãåāăą]", "a"), (r"[çćč]", "c"),
        (r"[èéêëēĕėęě]", "e"), (r"[ìíîïīĭį]", "i"),
        (r"[ñńň]", "n"), (r"[òóôöõøōŏő]", "o"),
        (r"[ùúûüūŭůűų]", "u"), (r"[ýÿ]", "y"),
        (r"[śšş]", "s"), (r"[žźż]", "z"), (r"ł", "l"),
        (r"œ", "oe"), (r"æ", "ae"),
    ])
    for pattern, value in replacements:
        expr = expr.str.replace_all(pattern, value)
    return (
        expr.str.replace_all(r"[^\p{L}\p{N}]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def token_rows(frame: pl.LazyFrame, text_column: str, id_column: str,
               stopwords: list[str]) -> pl.LazyFrame:
    return (
        frame.select(
            id_column,
            pl.col("country").hash(seed=2026).alias("country_key"),
            normalized(text_column).str.split(" ").list.unique().alias("token"),
        )
        .explode("token")
        .filter(
            (pl.col("token").str.len_chars() >= 2)
            & (~pl.col("token").is_in(stopwords))
        )
    )


def add_normalized_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        normalized("business_name").alias("name_norm"),
        normalized("business_address").alias("address_norm"),
    )


def add_combined_word_text(frame: pl.DataFrame) -> pl.DataFrame:
    """Add field-namespaced exact word tokens for sparse retrieval."""
    if "name_norm" not in frame.columns or "address_norm" not in frame.columns:
        frame = add_normalized_columns(frame)
    return frame.with_columns(
        pl.concat_str(
            pl.lit("n_") + pl.col("name_norm").str.replace_all(" ", " n_"),
            pl.lit("a_") + pl.col("address_norm").str.replace_all(" ", " a_"),
            separator=" ",
        ).alias("retrieval_text")
    )
