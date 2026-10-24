# -*- coding: utf-8 -*-
"""
Rebuild Table 6 / S5 composition counts from deposited title-level files.

Inputs (paths relative to repository root):
  derived_data_SI/S5_retrieval_sensitivity_BA_titles.csv
  Optional: a frozen-corpus sentiment table with columns
            outlet, sentiment  (not redistributed here if it contains bodies)

This script verifies B−A counts and sentiment shares from the deposited
1,074-title file. Pooling with A requires the researcher’s licensed corpus A
sentiment labels (same DistilRoBERTa title model as the manuscript).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BA = ROOT / "derived_data_SI" / "S5_retrieval_sensitivity_BA_titles.csv"
OUT = ROOT / "derived_data_SI" / "S5_retrieval_sensitivity_composition_from_BA.csv"

# Manuscript Table 6 reference for set A (frozen corpus)
A_REF = {
    "articles": 3327,
    "CNN": 1618,
    "NYT": 806,
    "WP": 903,
    "negative": 1462,
    "neutral": 1147,
    "positive": 718,
}


def shares(n_out: dict, n_sent: dict, n: int) -> dict:
    return {
        "CNN_share": n_out.get("CNN", 0) / n,
        "NYT_share": n_out.get("NYT", 0) / n,
        "WP_share": n_out.get("WP", 0) / n,
        "negative_share": n_sent.get("negative", 0) / n,
        "neutral_share": n_sent.get("neutral", 0) / n,
        "positive_share": n_sent.get("positive", 0) / n,
    }


def main() -> None:
    ba = pd.read_csv(BA)
    assert len(ba) == 1074, len(ba)
    n_out = ba["outlet"].value_counts().to_dict()
    n_sent = ba["sentiment"].str.lower().value_counts().to_dict()
    row_ba = {
        "set": "B-A",
        "articles": len(ba),
        "CNN": int(n_out.get("CNN", 0)),
        "NYT": int(n_out.get("NYT", 0)),
        "WP": int(n_out.get("WP", 0)),
        "negative": int(n_sent.get("negative", 0)),
        "neutral": int(n_sent.get("neutral", 0)),
        "positive": int(n_sent.get("positive", 0)),
    }
    row_ba.update(shares(n_out, n_sent, len(ba)))

    row_a = {"set": "A", **A_REF}
    row_a.update(
        shares(
            {"CNN": A_REF["CNN"], "NYT": A_REF["NYT"], "WP": A_REF["WP"]},
            {
                "negative": A_REF["negative"],
                "neutral": A_REF["neutral"],
                "positive": A_REF["positive"],
            },
            A_REF["articles"],
        )
    )

    row_b = {
        "set": "B",
        "articles": row_a["articles"] + row_ba["articles"],
        "CNN": row_a["CNN"] + row_ba["CNN"],
        "NYT": row_a["NYT"] + row_ba["NYT"],
        "WP": row_a["WP"] + row_ba["WP"],
        "negative": row_a["negative"] + row_ba["negative"],
        "neutral": row_a["neutral"] + row_ba["neutral"],
        "positive": row_a["positive"] + row_ba["positive"],
    }
    row_b.update(
        shares(
            {"CNN": row_b["CNN"], "NYT": row_b["NYT"], "WP": row_b["WP"]},
            {
                "negative": row_b["negative"],
                "neutral": row_b["neutral"],
                "positive": row_b["positive"],
            },
            row_b["articles"],
        )
    )

    out = pd.DataFrame([row_a, row_ba, row_b])
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
