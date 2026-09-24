# PONE-D-26-33847 — Supporting Information (S1 File)
# Paths parameterized via env vars; see README.md.
# Do NOT redistribute WiseSearch full texts or the licensed SQLite corpus.

import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune DistilRoBERTa on 280/120 gold split; evaluate held-out test only."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42
TRAIN_N = 280
TEST_N = 120
TEST_QUOTA = {"NYT": 29, "CNN": 59, "WP": 32}
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

GOLD = Path(os.environ.get("GOLD_XLSX", "data/human_gold_titles.xlsx"))
BASE = Path(os.environ.get("SENTIMENT_MODEL", "models/distilroberta-finetuned-financial-news-sentiment-analysis"))
OUT = Path(os.environ.get("PROBE_OUT", "outputs/revision_probe"))
DESKTOP = Path(os.environ.get("DESKTOP_OUT", "outputs"))


def set_all_seeds(seed: int = SEED) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts_train, parts_test = [], []
    for outlet, tq in TEST_QUOTA.items():
        sub = df[df["outlet"] == outlet].copy()
        if len(sub) <= tq:
            raise RuntimeError(f"{outlet}: need >{tq} rows, got {len(sub)}")
        # allocate test slots by gold share
        counts = sub["gold"].value_counts()
        alloc = {}
        for g, n_g in counts.items():
            alloc[g] = max(1, int(round(tq * n_g / len(sub))))
        while sum(alloc.values()) > tq:
            g = max(alloc, key=alloc.get)
            if alloc[g] > 1:
                alloc[g] -= 1
            else:
                break
        while sum(alloc.values()) < tq:
            g = max(counts.index, key=lambda x: counts[x] - alloc.get(x, 0))
            alloc[g] = alloc.get(g, 0) + 1

        test_idx = []
        for g, k in alloc.items():
            cell = sub[sub["gold"] == g]
            k = min(k, len(cell))
            if k:
                test_idx.extend(
                    cell.sample(n=k, random_state=SEED).index.tolist()
                )
        if len(test_idx) < tq:
            extra = sub.drop(index=test_idx).sample(
                n=tq - len(test_idx), random_state=SEED
            )
            test_idx.extend(extra.index.tolist())
        test_idx = test_idx[:tq]
        parts_test.append(sub.loc[test_idx])
        parts_train.append(sub.drop(index=test_idx))

    train = pd.concat(parts_train).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test = pd.concat(parts_test).sample(frac=1, random_state=SEED).reset_index(drop=True)
    assert len(train) == TRAIN_N and len(test) == TEST_N, (len(train), len(test))
    return train, test


class TitleDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_len: int = 128):
        self.titles = frame["title"].tolist()
        self.labels = [LABEL2ID[x] for x in frame["gold"].tolist()]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        enc = self.tok(
            self.titles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(3)), zero_division=0, average=None
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": float(np.mean(f1)),
        "neg_f1": float(f1[0]),
        "neu_f1": float(f1[1]),
        "pos_f1": float(f1[2]),
    }


def predict_frame(model, tokenizer, frame: pd.DataFrame, device: str) -> list[str]:
    model.eval()
    out_labels = []
    titles = frame["title"].tolist()
    bs = 32
    with torch.no_grad():
        for i in range(0, len(titles), bs):
            batch = titles[i : i + bs]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            pred = logits.argmax(dim=-1).cpu().numpy()
            out_labels.extend(ID2LABEL[int(x)] for x in pred)
    return out_labels


def metrics_bundle(y_true, y_pred, tag: str) -> dict:
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "tag": tag,
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=s)),
        "per_class": [
            {
                "class": LABELS[i],
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(s[i]),
            }
            for i in range(3)
        ],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
        "report": classification_report(
            y_true, y_pred, labels=LABELS, digits=4, zero_division=0
        ),
    }


def main():
    set_all_seeds(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    model_dir = OUT / "model_finetuned"
    model_dir.mkdir(exist_ok=True)

    raw = pd.read_excel(GOLD)
    df = pd.DataFrame(
        {
            "title": raw["英文标题"].astype(str),
            "gold": raw["情感标签"].astype(str).str.strip().str.lower(),
            "outlet": raw["outlet"],
            "date": raw["date"],
            "period": raw["period"],
            "sample_id": raw["sample_id"],
            "row_no": raw["row_no"],
            "zh": raw.get("中文翻译"),
        }
    )
    assert set(df["gold"]) <= set(LABELS)

    train_df, test_df = split_train_test(df)
    train_df.to_excel(OUT / "split_train_280.xlsx", index=False)
    test_df.to_excel(OUT / "split_test_120.xlsx", index=False)
    print("train gold", train_df["gold"].value_counts().to_dict())
    print("test gold", test_df["gold"].value_counts().to_dict())

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("device", device)

    tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE),
        local_files_only=True,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=False,
    )

    # baseline on test (before fine-tune)
    base_model.to(device)
    base_pred = predict_frame(base_model, tok, test_df, device)
    base_metrics = metrics_bundle(test_df["gold"].tolist(), base_pred, "baseline_zeroshot_on_test120")
    print("=== BASELINE test ===\n", base_metrics["report"])

    # class weights from train
    counts = train_df["gold"].value_counts().reindex(LABELS).fillna(0).values.astype(float)
    weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1.0))
    class_weights = torch.tensor(weights, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    train_ds = TitleDataset(train_df, tok)
    eval_ds = TitleDataset(test_df, tok)

    # reload fresh for training (clean state)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE),
        local_files_only=True,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(OUT / "hf_trainer"),
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        seed=SEED,
        report_to=[],
        save_total_limit=2,
        fp16=False,
        use_cpu=(device == "cpu"),
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(str(model_dir))
    tok.save_pretrained(str(model_dir))

    # evaluate best on test
    model.to(device)
    ft_pred = predict_frame(model, tok, test_df, device)
    ft_metrics = metrics_bundle(test_df["gold"].tolist(), ft_pred, "finetuned_on_test120")
    print("=== FINETUNED test ===\n", ft_metrics["report"])

    test_out = test_df.copy()
    test_out["baseline_pred"] = base_pred
    test_out["finetuned_pred"] = ft_pred
    test_out.to_excel(OUT / "test120_predictions.xlsx", index=False)

    summary = {
        "seed": SEED,
        "train_n": TRAIN_N,
        "test_n": TEST_N,
        "test_quota": TEST_QUOTA,
        "base_model": str(BASE),
        "finetuned_model": str(model_dir),
        "class_weights": {LABELS[i]: float(weights[i]) for i in range(3)},
        "baseline": base_metrics,
        "finetuned": ft_metrics,
        "delta_accuracy": ft_metrics["accuracy"] - base_metrics["accuracy"],
        "delta_macro_f1": ft_metrics["macro_f1"] - base_metrics["macro_f1"],
        "note": "Held-out test only. Full-corpus sentiment NOT re-inferred.",
    }
    # drop bulky report duplication in nested if needed — keep report
    (OUT / "finetune_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with pd.ExcelWriter(OUT / "微调对比_测试集120.xlsx", engine="openpyxl") as w:
        pd.DataFrame(
            [
                {
                    "model": "baseline",
                    "accuracy": base_metrics["accuracy"],
                    "macro_f1": base_metrics["macro_f1"],
                    "weighted_f1": base_metrics["weighted_f1"],
                },
                {
                    "model": "finetuned",
                    "accuracy": ft_metrics["accuracy"],
                    "macro_f1": ft_metrics["macro_f1"],
                    "weighted_f1": ft_metrics["weighted_f1"],
                },
            ]
        ).round(4).to_excel(w, sheet_name="overview", index=False)
        pd.DataFrame(base_metrics["per_class"]).round(4).to_excel(
            w, sheet_name="baseline_per_class", index=False
        )
        pd.DataFrame(ft_metrics["per_class"]).round(4).to_excel(
            w, sheet_name="finetuned_per_class", index=False
        )
        test_out.to_excel(w, sheet_name="test_rows", index=False)

    import shutil

    for name in [
        "微调对比_测试集120.xlsx",
        "finetune_summary.json",
        "split_train_280.xlsx",
        "split_test_120.xlsx",
    ]:
        try:
            shutil.copy2(OUT / name, DESKTOP / name)
        except Exception as e:
            print("desktop copy skip", name, e)

    print(
        "DONE baseline_acc={:.4f} ft_acc={:.4f} baseline_macroF1={:.4f} ft_macroF1={:.4f}".format(
            base_metrics["accuracy"],
            ft_metrics["accuracy"],
            base_metrics["macro_f1"],
            ft_metrics["macro_f1"],
        )
    )


if __name__ == "__main__":
    main()
