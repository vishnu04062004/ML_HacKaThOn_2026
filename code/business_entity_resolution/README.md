# Business Entity Resolution Pipeline

This is a memory-bounded solution for the full multi-million-row challenge data. It uses only the supplied TSV files and performs no external entity lookup or data augmentation.

From the `student_resource/` directory:

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/run_pipeline.py --data-dir dataset --output-dir output
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

For the optional memory-heavy target-ID audit, append `--check-ids` to the validator command.

## Design

The pipeline retrieves in the target-to-reference direction. Source 1 is deduplicated and every labeled Source-2/3 target has at most one Source-1 owner, so each target searches only its top three possible owners. Names and addresses become field-namespaced normalized word tokens; tokens present in over 1% of a country partition are removed; and sparse TF-IDF top-N multiplication is run independently inside every country label. This supports France without a hard-coded country encoder.

The ownership model uses retrieval score/rank, score separation, exact values, word and number overlap, edit similarities, postcodes, missingness, and length agreement. LightGBM learns from 200,000 uniformly sampled target records (599,027 retrieved pairs). A held-out target split selects the probability threshold using precision-heavy F0.5. At inference, every target can be assigned to at most one Source-1 entity. Candidate pairs contain the exact top-N pairs scored by the model; final matches are a subset.

Intermediate Parquet pieces live under `output/.cache` and are removed after successful output assembly. `output/run_metrics.json` and `output/ownership_model.joblib` are diagnostics; only the two TSV files belong in the submission archive.
