# db_new shared testdata

This directory holds small deterministic vector fixtures for db_new unit tests.

The default checked-in fixture is intentionally small for fast tests and easy
review. Generate a larger 128-dimensional corpus, for example 10k base vectors,
with:

```sh
python3 tests/db_new/shared/testdata/generate_vector_corpus.py \
  --base-count 10000 \
  --query-count 100 \
  --top-k 10 \
  --out-dir tests/db_new/shared/testdata
```

Fixture files use whitespace-separated TSV-like rows:

- `dense_128_base_<N>.tsv`: `doc_id v0 v1 ... v127`
- `dense_128_query_<Q>.tsv`: `query_id target_doc_id v0 v1 ... v127`
- `dense_128_groundtruth_<Q>x<K>.tsv`:
  `query_id target_doc_id neighbor0 ... neighbor<K-1>`
