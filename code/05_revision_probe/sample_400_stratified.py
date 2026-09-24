# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stratified sample of 400 titles for DistilRoBERTa fine-tune + held-out test.

Rules (frozen):
- Frame: cleaned corpus N≈3327 (NYT 806 / CNN 1618 / WP 903) from quchong保护0 - 副本.db
- Seed: 42
- Outlet: proportional to corpus size → NYT 97 / CNN 195 / WP 108 (=400)
- Period nested in outlet: 2013–2017 / 2018–2020 / 2021–2024, proportional within outlet
- Within each outlet×period cell: simple random sample (seed 42)
- If a cell is short: take all, redistribute remainder to other cells in same outlet
- Model labels (if available) are saved ONLY in the admin file, never on the coder sheet
- After human gold is complete: split 280 train / 120 test, stratified by outlet + gold
  (see split_train_test_after_gold below; do NOT split before labeling)

Outputs (Desktop):
- 标注表_微调400_编码用.xlsx   — blank gold column for the coder
- 标注表_微调400_管理用.xlsx   — same IDs + sampling strata (+ model_* if present)
- sample_400_manifest.json  — frozen IDs and rules
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

SEED = 42
N_TOTAL = 400
# proportional to 806 : 1618 : 903
OUTLET_QUOTA = {"NYT": 97, "CNN": 195, "WP": 108}
PERIOD_BINS = [
    ("2013-2017", 2013, 2017),
    ("2018-2020", 2018, 2020),
    ("2021-2024", 2021, 2024),
]
TRAIN_N = 280
TEST_N = 120

DB_CANDIDATES = [
    Path(os.environ.get("CORPUS_DB", "data/corpus.db")),
    Path(os.environ.get("CORPUS_DB", "data/corpus.db")),
    Path(os.environ.get("CORPUS_DB", "data/corpus.db")),
]
PRED_CANDIDATES = [
    Path("/Volumes/KESU/研究生作业/新建文件夹/情绪分析集成.csv"),
    Path("/Volumes/KESU/新建文件夹/情绪分析集成.csv"),
]
OUT_DIR = Path("/Users/chaos/cursor_use/PLOS_返修材料/sample_400")
DESKTOP_DIR = Path(os.environ.get("DESKTOP_OUT", "outputs"))


def find_db() -> Path:
    for p in DB_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "KESU DB not found. Remount the drive, then re-run this script.\n"
        + "\n".join(str(p) for p in DB_CANDIDATES)
    )


def parse_year(val) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    m = re.search(r"(20\d{2})", s)
    return int(m.group(1)) if m else None


def period_of(year: int | None) -> str | None:
    if year is None:
        return None
    for name, lo, hi in PERIOD_BINS:
        if lo <= year <= hi:
            return name
    return None


def load_corpus(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    frames = []
    # Known schema for quchong保护0 - 副本.db:
    #   news=CNN(1618), wpnews=WP(903), my_existing_table=NYT(806)
    explicit = {
        "news": "CNN",
        "wpnews": "WP",
        "my_existing_table": "NYT",
    }
    mapping = []
    for t in tables:
        if t in explicit:
            mapping.append((t, explicit[t]))
            continue
        tl = t.lower()
        if "nyt" in tl or "纽约" in t:
            mapping.append((t, "NYT"))
        elif "cnn" in tl:
            mapping.append((t, "CNN"))
        elif "wp" in tl or "华盛顿" in t or "washington" in tl:
            mapping.append((t, "WP"))
    # de-dupe by outlet (prefer explicit)
    seen = {}
    for t, o in mapping:
        if o not in seen:
            seen[o] = t
    mapping = [(t, o) for o, t in seen.items()]
    if len(mapping) < 3:
        raise RuntimeError(f"Could not map outlet tables in {db_path}: {tables}")

    for table, outlet in mapping:
        cols_info = [c[1] for c in conn.execute(f"PRAGMA table_info([{table}])")]
        cols = {c.lower(): c for c in cols_info}
        title_c = cols.get("title")
        date_c = cols.get("date")
        if not title_c:
            raise RuntimeError(f"{table}: no title column ({cols_info})")
        # titles only — skip body (large) for sampling speed on external drive
        q = f"SELECT [{title_c}] AS title, [{date_c}] AS date FROM [{table}]"
        df = pd.read_sql_query(q, conn)
        out = pd.DataFrame(
            {
                "outlet": outlet,
                "title": df["title"].astype(str),
                "date": df["date"],
            }
        )
        frames.append(out)
    conn.close()
    all_df = pd.concat(frames, ignore_index=True)
    all_df["year"] = all_df["date"].map(parse_year)
    all_df["period"] = all_df["year"].map(period_of)
    all_df = all_df[all_df["title"].str.strip().ne("") & all_df["title"].ne("nan")]
    all_df = all_df[all_df["period"].notna()].copy()
    all_df["sample_id"] = (
        all_df["outlet"]
        + "_"
        + all_df.groupby("outlet").cumcount().astype(str).str.zfill(4)
    )
    return all_df.reset_index(drop=True)


def proportional_quota(n_cell: int, n_group: int, group_quota: int) -> int:
    if n_group <= 0:
        return 0
    return int(round(group_quota * n_cell / n_group))


def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    rng = pd.Series(range(len(df)), index=df.index)  # noqa: for clarity
    picked = []
    for outlet, oq in OUTLET_QUOTA.items():
        sub = df[df["outlet"] == outlet]
        if len(sub) < oq:
            raise RuntimeError(f"{outlet}: only {len(sub)} rows, need {oq}")
        # period quotas proportional within outlet
        period_counts = sub["period"].value_counts()
        raw = {
            p: proportional_quota(int(period_counts.get(p, 0)), len(sub), oq)
            for p, _, _ in PERIOD_BINS
        }
        # fix rounding to exact oq
        diff = oq - sum(raw.values())
        # adjust largest period
        order = sorted(raw.keys(), key=lambda p: -period_counts.get(p, 0))
        i = 0
        while diff != 0 and order:
            p = order[i % len(order)]
            if diff > 0 and period_counts.get(p, 0) > raw[p]:
                raw[p] += 1
                diff -= 1
            elif diff < 0 and raw[p] > 0:
                raw[p] -= 1
                diff += 1
            else:
                i += 1
                if i > 100:
                    break
                continue
            i += 1

        taken_idx = []
        shortfall = 0
        for p, q in raw.items():
            cell = sub[sub["period"] == p]
            k = min(q, len(cell))
            shortfall += q - k
            if k > 0:
                taken_idx.extend(cell.sample(n=k, random_state=SEED).index.tolist())

        # redistribute shortfall from remaining rows in outlet
        if shortfall > 0:
            remain = sub.drop(index=taken_idx)
            extra = remain.sample(n=min(shortfall, len(remain)), random_state=SEED)
            taken_idx.extend(extra.index.tolist())

        # if over (shouldn't), trim
        if len(taken_idx) > oq:
            taken_idx = (
                pd.Series(taken_idx)
                .sample(n=oq, random_state=SEED)
                .tolist()
            )
        picked.extend(taken_idx)

    out = df.loc[picked].copy()
    out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out["row_no"] = range(1, len(out) + 1)
    assert len(out) == N_TOTAL, len(out)
    return out


def try_attach_predictions(sample: pd.DataFrame) -> pd.DataFrame:
    for p in PRED_CANDIDATES:
        if not p.exists():
            continue
        pred = pd.read_csv(p)
        # best-effort title join
        tcol = next((c for c in pred.columns if c.lower() == "title"), None)
        if not tcol:
            continue
        pred = pred.copy()
        pred["_t"] = pred[tcol].astype(str).str.strip().str.lower()
        sample = sample.copy()
        sample["_t"] = sample["title"].astype(str).str.strip().str.lower()
        label_cols = [
            c
            for c in pred.columns
            if any(
                k in c.lower()
                for k in ("label", "sentiment", "pred", "prob", "score")
            )
        ]
        keep = ["_t"] + label_cols
        merged = sample.merge(
            pred[keep].drop_duplicates("_t"), on="_t", how="left", suffixes=("", "_m")
        )
        return merged.drop(columns=["_t"])
    return sample


def export_xlsx(sample: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coder_path = OUT_DIR / "标注表_微调400_编码用.xlsx"
    admin_path = OUT_DIR / "标注表_微调400_管理用.xlsx"
    manifest_path = OUT_DIR / "sample_400_manifest.json"
    # best-effort mirror to Desktop
    try:
        DESKTOP_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    coder = pd.DataFrame(
        {
            "row_no": sample["row_no"],
            "sample_id": sample["sample_id"],
            "outlet": sample["outlet"],
            "date": sample["date"],
            "period": sample["period"],
            "title": sample["title"],
            "gold": "",  # positive / neutral / negative
            "notes": "",
        }
    )
    rules = pd.DataFrame(
        {
            "rule": [
                "只根据 TITLE 标注，不要打开全文改标签。",
                "标签只能是：positive / neutral / negative（小写英文）。",
                "positive：标题本身对经济表现、协议、复苏等有明确好评。",
                "negative：标题本身有明确差评、损失、冲突、危机等。",
                "neutral：陈述政策/事件，标题本身无明显评价方向。",
                "tariff/sanctions/restrictions/investigation 若只是陈述政策、无额外贬义 → neutral。",
                "不确定时标 neutral，并在 notes 写一句原因。",
                "不要查看模型预测；本表故意不含 model_label。",
                f"本批共 {N_TOTAL} 条；标完后按 outlet+gold 分层切 {TRAIN_N} 训练 / {TEST_N} 测试。",
                f"随机种子 SEED={SEED}；媒体配额 NYT97 / CNN195 / WP108；时段 2013-17 / 2018-20 / 2021-24。",
            ]
        }
    )
    with pd.ExcelWriter(coder_path, engine="openpyxl") as w:
        coder.to_excel(w, sheet_name="annotate", index=False)
        rules.to_excel(w, sheet_name="coding_rules", index=False)

    sample.to_excel(admin_path, index=False)

    manifest = {
        "seed": SEED,
        "n_total": N_TOTAL,
        "outlet_quota": OUTLET_QUOTA,
        "period_bins": [b[0] for b in PERIOD_BINS],
        "train_n": TRAIN_N,
        "test_n": TEST_N,
        "split_rule": "After gold labeling: stratified 280/120 by outlet + gold (seed 42).",
        "sample_ids": sample["sample_id"].tolist(),
        "outlet_counts": sample["outlet"].value_counts().to_dict(),
        "period_counts": sample["period"].value_counts().to_dict(),
        "crosstab": (
            sample.groupby(["outlet", "period"]).size().astype(int).to_dict()
        ),
    }
    # json-serialize tuple keys
    manifest["crosstab"] = {
        f"{a}|{b}": int(v) for (a, b), v in sample.groupby(["outlet", "period"]).size().items()
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # mirror to Desktop when permitted
    import shutil

    for src in (coder_path, admin_path, manifest_path):
        try:
            shutil.copy2(src, DESKTOP_DIR / src.name)
        except Exception as e:
            print("Desktop copy skipped:", src.name, e)
    print("Wrote", coder_path)
    print("Wrote", admin_path)
    print("Wrote", manifest_path)
    print("outlet:", sample["outlet"].value_counts().to_dict())
    print("period:", sample["period"].value_counts().to_dict())


def split_train_test_after_gold(labeled_xlsx: Path) -> None:
    """Call after gold is filled. Expects columns sample_id, outlet, gold, title."""
    df = pd.read_excel(labeled_xlsx)
    df["gold"] = df["gold"].astype(str).str.strip().str.lower()
    ok = {"positive", "neutral", "negative"}
    bad = df[~df["gold"].isin(ok)]
    if len(bad):
        raise ValueError(f"{len(bad)} rows missing/invalid gold; fix before split.")
    assert len(df) == N_TOTAL
    parts = []
    # proportional test slots by outlet
    test_quota = {"NYT": 29, "CNN": 59, "WP": 32}  # sums 120
    for outlet, tq in test_quota.items():
        sub = df[df["outlet"] == outlet]
        # within outlet, stratified by gold
        test_idx = []
        remain_q = tq
        labels = list(sub["gold"].value_counts().index)
        # allocate by gold share
        alloc = {}
        for g in labels:
            n_g = (sub["gold"] == g).sum()
            alloc[g] = max(1, int(round(tq * n_g / len(sub)))) if n_g else 0
        # fix sum
        while sum(alloc.values()) > tq:
            g = max(alloc, key=alloc.get)
            if alloc[g] > 1:
                alloc[g] -= 1
            else:
                break
        while sum(alloc.values()) < tq:
            g = max(labels, key=lambda x: (sub["gold"] == x).sum() - alloc.get(x, 0))
            alloc[g] = alloc.get(g, 0) + 1
        for g, k in alloc.items():
            cell = sub[sub["gold"] == g]
            k = min(k, len(cell))
            test_idx.extend(cell.sample(n=k, random_state=SEED).index.tolist())
        # top up
        if len(test_idx) < tq:
            extra = sub.drop(index=test_idx).sample(
                n=tq - len(test_idx), random_state=SEED
            )
            test_idx.extend(extra.index.tolist())
        test_idx = test_idx[:tq]
        test = sub.loc[test_idx].copy()
        train = sub.drop(index=test_idx).copy()
        test["split"] = "test"
        train["split"] = "train"
        parts.extend([train, test])
    out = pd.concat(parts, ignore_index=True)
    assert (out["split"] == "train").sum() == TRAIN_N
    assert (out["split"] == "test").sum() == TEST_N
    out_path = OUT_DIR / "标注表_微调400_train_test.xlsx"
    out.to_excel(out_path, index=False)
    print("Wrote", out_path)


def main():
    db = find_db()
    print("DB:", db)
    corpus = load_corpus(db)
    print("corpus", len(corpus), corpus["outlet"].value_counts().to_dict())
    sample = stratified_sample(corpus)
    sample = try_attach_predictions(sample)
    export_xlsx(sample)


if __name__ == "__main__":
    main()
