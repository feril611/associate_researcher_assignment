#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "training-dataset.csv"  # relative to src/
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_TRAIN = 8
BATCH_VAL = 4
LR = 2e-3
WEIGHT_DECAY = 0.01
EPOCHS = 2
N_SPLITS = 5
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EDataset(Dataset):
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = np.asarray(labels, dtype=np.int64).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.enc["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.enc["attention_mask"][idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, steps = 0.0, 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        steps += 1
        preds = out.logits.argmax(dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return {
        "loss": total_loss / max(1, steps),
        "accuracy": acc,
        "f1": f1,
        "y_true": all_labels,
        "y_pred": all_preds
    }

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    df = pd.read_csv(DATA_PATH, encoding="utf-8")[["text", "label"]].dropna()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_acc, fold_f1 = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df["text"], df["label"]), start=1):
        print(f"\n----- Fold {fold}/{N_SPLITS} -----")

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)

        train_enc = tokenizer(
            train_df["text"].astype(str).tolist(),
            truncation=True, padding="max_length", max_length=MAX_LEN
        )
        val_enc = tokenizer(
            val_df["text"].astype(str).tolist(),
            truncation=True, padding="max_length", max_length=MAX_LEN
        )

        train_ds = EDataset(train_enc, train_df["label"])
        val_ds = EDataset(val_enc, val_df["label"])
        train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            train_preds, train_labels = [], []

            for step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                out = model(**batch)
                loss = out.loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = out.logits.argmax(dim=-1).cpu().tolist()
                labels = batch["labels"].cpu().tolist()
                train_preds.extend(preds)
                train_labels.extend(labels)

                if step % 5 == 0 or step == len(train_loader):
                    print(f"[Fold {fold} | Epoch {epoch} | Step {step}/{len(train_loader)}] "
                          f"Avg Train Loss {running_loss/step:.4f}")

            tr_acc = accuracy_score(train_labels, train_preds)
            tr_f1 = f1_score(train_labels, train_preds, average="weighted")
            val = evaluate(model, val_loader, device)
            print(f"[Fold {fold} | Epoch {epoch}] "
                  f"Train Acc {tr_acc:.4f} | Train F1 {tr_f1:.4f} || "
                  f"Val Loss {val['loss']:.4f} | Val Acc {val['accuracy']:.4f} | Val F1 {val['f1']:.4f}")

        # record fold results
        fold_acc.append(val["accuracy"]); fold_f1.append(val["f1"])
        cm = confusion_matrix(val["y_true"], val["y_pred"], labels=[0,1])
        print(f"[Fold {fold}] Confusion Matrix:\n{cm}")


if __name__ == "__main__":
    main()
