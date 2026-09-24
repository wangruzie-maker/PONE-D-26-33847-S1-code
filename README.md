# PONE-D-26-33847 — Supporting Information (S1 File): Analysis Code

**Manuscript:** A computational study of sentiment and thematic structure in U.S. news coverage of China’s economy

Author-generated Python scripts for corpus cleaning, DistilRoBERTa title-level sentiment scoring, BERTopic thematic modeling (outlet-specific UMAP/HDBSCAN / Optuna), figure helpers, and the revision-stage stratified human-coded sentiment probe (N=400; holdout accuracy 0.80; macro-F1 0.80).

## Not included (do not redistribute)

- WiseSearch full-text SQLite corpus (`corpus.db`) — copyrighted third-party content under license
- Local model weight folders (download DistilRoBERTa financial-news sentiment and `all-MiniLM-L6-v2` yourself)
- Unrelated scripts (ModelScope download) were omitted; adjacent-experiment scripts sit under `code/archive_exploratory/`

## Layout

```
code/
  01_data_cleaning/
  02_sentiment/
  03_bertopic/
  04_figures_tables/
  05_revision_probe/
  archive_exploratory/
  config_template.env
derived_data_SI/          # title-level / aggregate tables only
requirements.txt
README.md
README_提交说明_中文.md
```

## Environment

1. Place a licensed local corpus DB at `data/corpus.db` (or set `CORPUS_DB`).
2. Place model weights under `models/` (or set `SENTIMENT_MODEL` / `EMBEDDING_MODEL`).
3. `pip install -r requirements.txt`
4. Export variables from `code/config_template.env`.

### BERTopic parameters in the manuscript

| Outlet | n_neighbors | n_components | min_dist | min_cluster_size |
|--------|-------------|--------------|----------|------------------|
| New York Times | 16 | 10 | 0.27 | 13 |
| CNN | 23 | 6 | 0.68 | 7 |
| Washington Post | 6 | 5 | 0.02 | 21 |

Seed used in Optuna / sampling scripts: **42**.

### Revision probe (Section 3.2)

Stratified **400** titles; holdout **120**: accuracy **0.80**, macro-F1 **0.80**. Main corpus labels remain the original DistilRoBERTa outputs; the probe is a sample-level accuracy check.

## Derived data SI

`derived_data_SI/` holds monthly counts, human-probe title tables, and holdout metrics — **no article bodies**.

## Reproducibility note

Scripts come from the original Windows analysis environment; drive letters were replaced with environment variables. Re-fitting BERTopic with newer package versions may change topic IDs; primary deposited outputs are the derived tables.
