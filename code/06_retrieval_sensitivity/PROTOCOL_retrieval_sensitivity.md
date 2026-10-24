# Retrieval-sensitivity screen (Section 4.4 / Table 6)

This folder documents how the **1,074** added titles (set **B−A**) were obtained so that other researchers with WiseSearch access can repeat the operation.

## Purpose

Test whether conflict-oriented terms in the original title-field query (Section 3.1) manufacture the aggregate negative tilt. The main analyses remain on the frozen corpus **A** (*n* = 3,327). The screen is a sensitivity check only; BERTopic domains were **not** re-estimated for B−A.

## Retrieval (reproduce with WiseSearch)

| Setting | Value |
|--------|--------|
| Database / platform | WiseSearch (institutional subscription) |
| Export date used in the paper | **18 October 2026** |
| Field | **Title** (same as Section 3.1) |
| Outlets | CNN; The New York Times; The Washington Post |
| Period | **2013-09-01** to **2024-12-31** (same as Section 3.1) |
| Original query (set A) | Exact-phrase title query Q0 as reported in Section 3.1 |
| Expanded query (set B) | Same field / outlets / period, with four reviewer-suggested phrases added by **OR**: `"China investment"` OR `"China exports"` OR `"China economic cooperation"` OR `"China GDP"` |

Researchers reconstruct **B** by re-running the expanded title query under the same constraints, then derive **B−A** by deduplication against A (below).

## Cleaning

Apply the **same cleaning procedures as Section 3.1** (outlet filter, study-window filter, title normalization, within-export deduplication). Do not redistribute WiseSearch full texts.

## Deduplication → 1,074 (B−A)

1. Build match keys for frozen corpus A: `(outlet, date, normalized_title)`.
2. Normalization used in this deposit (see deposited `title_key` column):
   - lowercase;
   - unify curly quotes;
   - strip trailing outlet suffixes such as `- The New York Times` / `- The Washington Post`;
   - drop text after `|`;
   - keep alphanumeric tokens only; collapse whitespace.
3. Remove any expanded-export title whose key is already in A.
4. Retain economically relevant China titles consistent with the paper’s screen.
5. Resulting **B−A** counts: **1,074** titles (**CNN 633**; **NYT 110**; **WP 331**).

Deposited file: `derived_data_SI/S5_retrieval_sensitivity_BA_titles.csv`  
Columns: `date`, `outlet`, `title`, `title_key`, `sentiment`, `batch`  
(`batch` marks export batches used when assembling CNN vs NYT/WP supplements; it is not a manuscript table column.)

## Sentiment scoring

- Model: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` (same DistilRoBERTa financial-news model as the main analyses).
- Input: **title text only**.
- Labels: negative / neutral / positive (as deposited in `sentiment`).

## Composition table (Table 6)

| Set | Definition | *n* |
|-----|------------|-----|
| **A** | Frozen cleaned corpus | 3,327 |
| **B−A** | Added titles after dedupe vs A | 1,074 |
| **B** | A ∪ (B−A) | 4,401 |

Deposited summary: `derived_data_SI/S5_retrieval_sensitivity_composition.csv`  

Headline result used in the paper:

- B−A sentiment: **38.4%** negative, **40.6%** neutral, **21.0%** positive (less negative than A).
- After pooling (B): negative share moves from **43.9%** (A) to **42.6%** (B); positive share stays near **21.6%**.

Rebuild B−A counts from the deposited titles:

```bash
python code/06_retrieval_sensitivity/reproduce_composition_table.py
```

## What was not done

- No BERTopic re-fit on B−A; four thematic-domain shares were **not** re-estimated for the added titles.
- Main-corpus DistilRoBERTa labels were **not** replaced by this screen.
