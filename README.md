# PONE-D-26-33847 — Supporting Information (S1 File): Analysis Code & Derived Data

**Manuscript:** A computational study of sentiment and thematic structure in U.S. news coverage of China’s economy  
**Repository:** author-generated scripts and **title-level / aggregate** derived tables only.

This README is the English deposit note for reproducing the revision-stage additions: (1) the stratified DistilRoBERTa human probe (Section 3.2 / Table 1), and (2) the WiseSearch retrieval-sensitivity screen that yields **1,074** added titles (Section 4.4 / Table 6).

---

## What is and is not included

### Included

| Path | Contents |
|------|----------|
| `code/01_data_cleaning/` | Parse / clean / deduplicate WiseSearch exports (local DB workflow) |
| `code/02_sentiment/` | DistilRoBERTa title inference; VADER lexicon check |
| `code/03_bertopic/` | Outlet-specific BERTopic / UMAP–HDBSCAN / Optuna search |
| `code/04_figures_tables/` | Sentiment distribution and time-series helpers |
| `code/05_revision_probe/` | Stratified sample of 400 titles; train-280 / test-120 holdout eval |
| `code/06_retrieval_sensitivity/` | Protocol + script to rebuild Table 6 composition from deposited B−A titles |
| `derived_data_SI/` | Monthly counts; human-probe tables; holdout metrics; **S5 retrieval-sensitivity** titles |

### Not included (do not redistribute)

- WiseSearch full-text SQLite corpus (`corpus.db`) — copyrighted third-party content under license
- Local model weight folders (download DistilRoBERTa financial-news sentiment and `all-MiniLM-L6-v2` yourself)
- Unrelated scripts (omitted or under `code/archive_exploratory/`)

---

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

---

## A. Human-coded sentiment probe (Section 3.2 / Table 1)

### Design

- Stratified gold set of **400** titles (NYT 97, CNN 195, WP 108).
- **Coding rules:** negative = risk, loss, conflict, decline, or threat; positive = growth, cooperation, recovery, or gain; neutral = factual or policy text without clear evaluation.
- **Coders:** one primary coder labeled all 400; a second coder double-coded 100 titles (**κ = 0.74**).
- **Split:** **280** titles for validation-only training adaptation of DistilRoBERTa; **120** held out for final evaluation. The 280 were **not** used to rewrite coding rules or to re-label the main corpus.
- Main-analysis labels remain the original DistilRoBERTa title outputs; this probe is a **sample-level** check (Limitations: not full-corpus re-inference).

### Holdout performance (n = 120) — manuscript Table 1

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Negative | 0.895 | 0.810 | 0.850 | 63 |
| Neutral | 0.619 | 0.788 | 0.693 | 33 |
| Positive | 0.905 | 0.792 | 0.844 | 24 |
| Overall | — | — | macro-F1 **0.796** | Accuracy **0.800** |

Negative precision of 0.895 does not indicate systematic over-calling of negativity; the main confusion was between negative and neutral.

### Deposited probe files (`derived_data_SI/`)

| File | Description |
|------|-------------|
| `S4_human_probe_400_titles_vs_model.xlsx` | 400-title probe vs model |
| `S4_split_train_280.xlsx` / `S4_split_test_120.xlsx` | Train / holdout splits |
| `S4_holdout_FINAL_metrics.json` | Holdout accuracy, macro-F1, per-class P/R/F1, confusion matrix |
| `S4_human_probe_metrics_summary.xlsx` / `S4_human_probe_summary.json` | Additional probe summaries |
| `S4_sample_400_manifest.json` | Sampling manifest |

Scripts: `code/05_revision_probe/sample_400_stratified.py`, `probe_finetune_holdout_eval.py`.

---

## B. Retrieval-sensitivity screen — 1,074 added titles (Section 4.4 / Table 6)

Full step-by-step protocol: [`code/06_retrieval_sensitivity/PROTOCOL_retrieval_sensitivity.md`](code/06_retrieval_sensitivity/PROTOCOL_retrieval_sensitivity.md).

### How B−A was obtained (repeatable with WiseSearch)

1. **Same** title field, outlets (CNN / NYT / WP), and study period (**2013-09-01**–**2024-12-31**) as Section 3.1.
2. Expanded retrieval (export date in the paper: **18 October 2026**) adds four reviewer-suggested phrases by **OR**:  
   `"China investment"` OR `"China exports"` OR `"China economic cooperation"` OR `"China GDP"`.
3. **Same cleaning** as Section 3.1.
4. **Deduplicate** against frozen corpus A (*n* = 3,327) on `(outlet, date, normalized_title)`.
5. Result: **B−A = 1,074** (CNN 633; NYT 110; WP 331). **B = A ∪ (B−A) = 4,401**.
6. Score B−A titles with the **same DistilRoBERTa** financial-news model (titles only). Do **not** re-run BERTopic on B−A.

### Headline composition (manuscript Table 6)

| Set | Articles | Neg. share | Neu. share | Pos. share |
|-----|----------|------------|------------|------------|
| A (frozen) | 3,327 | 43.9% | 34.5% | 21.6% |
| B−A (added) | 1,074 | 38.4% | 40.6% | 21.0% |
| B (pooled) | 4,401 | 42.6% | 36.0% | 21.5% |

Pooling moves the negative share only from 43.9% to 42.6%; the original query does not manufacture the aggregate negative tilt on this screen.

### Deposited sensitivity files (`derived_data_SI/`)

| File | Description |
|------|-------------|
| `S5_retrieval_sensitivity_BA_titles.csv` | All **1,074** B−A titles + DistilRoBERTa labels (no bodies) |
| `S5_retrieval_sensitivity_composition.csv` | A / B−A / B composition table (Table 6 source) |

Rebuild composition counts from the deposited B−A file:

```bash
python code/06_retrieval_sensitivity/reproduce_composition_table.py
```

---

## Layout

```
code/
  01_data_cleaning/
  02_sentiment/
  03_bertopic/
  04_figures_tables/
  05_revision_probe/
  06_retrieval_sensitivity/   # NEW: protocol + Table 6 rebuild
  archive_exploratory/
  config_template.env
derived_data_SI/              # title-level / aggregate tables only (incl. S4, S5)
requirements.txt
README.md
```

---

## Reproducibility notes

- Scripts come from the original Windows analysis environment; drive letters were replaced with environment variables where practical.
- Re-fitting BERTopic with newer package versions may change topic IDs; primary deposited outputs are the derived tables.
- Outlet-specific final UMAP/HDBSCAN parameters are those in the manuscript table above; search scripts accept CLI args rather than hard-coding one driver for all three outlets.
- Probability-weighted / high-confidence robustness tables and some figure assembly steps used interactive CSV exports that are only partially reflected as scripts.
- Scripts are research prototypes (hard-coded table names, mixed comments); they document the methods used, not a polished package API.

## Coverage vs manuscript pipeline

**Included:** DistilRoBERTa title inference; VADER check; BERTopic fit / visualization / Optuna search; basic sentiment plots; revision probe (400 / 280 / 120); retrieval-sensitivity **protocol + deposited B−A titles + composition table**.

**Not a one-click reproduction of every manuscript table/figure:** WiseSearch re-query requires institutional access; full-text DB is not deposited; thematic-domain shares for B−A were intentionally not re-estimated.
