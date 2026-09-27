# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Fill before final submission]  
**Team Members:** [Fill before final submission]  
**Submission Date:** 25 September 2026

---

## 1. Executive Summary

We use a memory-bounded hybrid entity-resolution system: open-country text normalization, reverse sparse TF-IDF retrieval from each Source-2/3 record to the deduplicated Source-1 reference, and a LightGBM ownership classifier with precision-heavy threshold calibration. The key structural innovation is exploiting the verified training invariant that each target record has at most one Source-1 owner, while each Source-1 entity may own many target records.

## 2. Methodology

### 2.1 Problem Analysis

The official data contains 2,206,821 training Source-1 entities, 10,320,219 training target records, 1,732,544 test Source-1 entities, and 9,969,589 test targets. Training has 7,638,365 positive links and 123,247 singleton Source-1 entities. Every labeled target appears under exactly one Source-1 entity; no target has duplicate owners.

Names exhibit legal-suffix changes, punctuation, word reordering, typos, Indic scripts/transliteration, and trade-name noise. Addresses exhibit reordered components, abbreviations, partial values, landmarks, and missingness (344,883 training target addresses are empty). The unseen France partition requires treating country as an open string label.

### 2.2 Solution Strategy

**Approach Type:** Sparse blocking + supervised ownership classifier + global uniqueness constraint  
**Core Innovation:** Reverse target-to-reference retrieval. Each target searches for its three best Source-1 owners, after which the model assigns it to zero or one owner. Predictions are inverted at the end to create each Source-1 entity's one-to-many target list.

Normalization lowercases text, folds common Latin diacritics without damaging Indic scripts, standardizes punctuation/whitespace, and preserves separate name/address namespaces. Retrieval is performed independently within every observed country string, including France; there is no fixed country enumeration or country one-hot feature.

## 3. Candidate Generation (Blocking)

Names and addresses become field-prefixed word tokens (`n_...` and `a_...`). A country-local TF-IDF index is fitted on Source 1 and queried by Source-2/3 targets using `sparse-dot-topn`. Tokens appearing in more than 1% of a country's Source-1 records are removed because they add sparse-intersection cost but almost no discriminative information. The three highest-cosine Source-1 owners per target form the exact candidate set passed to the classifier.

- **Blocking keys used:** normalized field-aware TF-IDF words, country equality, top-3 sparse cosine retrieval.
- **Final test candidate pairs:** 29,839,707.
- **Sampled training top-3 link recall:** 0.939830.
- **Reduction ratio:** over 99.9998% versus approximately 17.3 trillion unrestricted test comparisons.
- **Recall protection:** independent name/address tokens, diacritic folding, full target-to-reference search within country, and three owners retained per target.

## 4. Matching Model

**Features used:**

- Retrieval: TF-IDF cosine, retrieval rank, distance/ratio to the best owner.
- Name: token Jaccard/containment, exact equality, length ratio, RapidFuzz edit ratio and weighted ratio.
- Address: token Jaccard/containment, exact equality, length ratio, RapidFuzz weighted ratio, missingness.
- Structure: numeric-token agreement, postcode agreement/conflict, target source.

**Model type:** LightGBM gradient-boosted trees (MIT license), 450 trees, 31 leaves.  
**Training:** 200,000 uniformly sampled Source-2/3 targets produced 599,027 candidate pairs and 139,155 retrieved positives.  
**Threshold selection:** deterministic held-out target split, direct F0.5 grid optimization, conservative tie-breaking. A target is assigned only to its highest-probability candidate and only above the learned threshold (0.800909), enforcing at most one owner per target.

## 5. Results & Error Analysis

The held-out target-assignment proxy (including unmatched target records) achieved:

- **F0.5:** 0.947323
- **Precision:** 0.978464
- **Recall:** 0.840342
- **Top-3 blocking recall:** 0.939830

This is a target-assignment validation score rather than the hidden leaderboard's exact per-Source-1 macro score; the official macro score cannot be computed for test data. The final test output predicts 5,192,197 links and 124,479 Source-1 singletons. Country-level mean predicted links are France 3.049, India 2.896, and US 3.099, with no anomalous collapse on unseen France.

Common false positives involve short/generic business names combined with common or partial addresses. Common false negatives involve cross-script transliteration, DBA names with no preserved distinctive token, heavily corrupted short names, and missing target addresses. High-frequency token pruning reduced sampled blocking recall by about two points but changed held-out F0.5 by only 0.00054 while eliminating memory failures and improving full inference throughput by more than an order of magnitude.

## 6. Conclusion

Reverse retrieval aligns the algorithm with the dataset's ownership structure and makes nearly ten million target records tractable on a 16 GB machine. Sparse field-aware blocking, nonlinear lexical/structural scoring, calibrated precision, and unique target ownership provide a strong, reproducible solution while using only supplied data.

## Appendix

### A. Code Artefacts

Entry point: `code/business_entity_resolution/src/run_pipeline.py`. Supporting modules under `src/er_pipeline/` implement normalization, sparse retrieval, pair features, modeling, output assembly, and diagnostics. Exact commands and pinned dependencies are in `code/business_entity_resolution/README.md` and `requirements.txt`.

### B. Additional Results

The final official validator result is `PASS`. Both TSV files contain exactly 1,732,544 data rows. `matching_results.tsv` is 89,377,253 bytes and `candidate_pairs.tsv` is 406,932,992 bytes. Final matches are a strict subset of candidates by construction and by validator check.
