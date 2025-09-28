# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import List, Dict

from GE2PE import GE2PE

DATA_PATH = r"D:/Homograph/Data/Data/Benchmark_Dataset.jsonl"   
USE_RULES = True                               
SHOW_MISMATCH_EXAMPLES = 5                     

MODELS = {
    "base-ge2pe":   r"D:/Taha/Homograph/ge2pe/base-ge2pe",
    "homo-ge2pe":   r"D:/Taha/Homograph/ge2pe/homo-ge2pe",
    "homo-t5":      r"D:/Taha/Homograph/ge2pe/homo-t5",
}

def load_jsonl(path: str) -> List[Dict]:

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                records.append(rec)
            except json.JSONDecodeError as e:
                raise ValueError(f"خط {ln} فایل JSON نیست: {e}")
    if not records:
        raise ValueError("file is empty")
    
    need_keys = {"Sentence", "Homograph_Phoneme_GE2PE"}
    miss = [i for i, r in enumerate(records) if not need_keys.issubset(r.keys())]
    if miss:
        raise KeyError(f"no record (Sentence, Homograph_Phoneme_GE2PE). : index {miss[:3]}")
    return records

def evaluate_model(model: GE2PE, sentences: List[str], labels: List[str], use_rules: bool = True):
  
    outputs = model.generate(sentences, use_rules=use_rules)  
    preds = [o if isinstance(o, str) else str(o) for o in outputs]

    flags = [lbl in pred for lbl, pred in zip(labels, preds)]
    correct = sum(flags)
    total = len(labels)
    acc = correct / total if total else 0.0

    mismatches = []
    for i, ok in enumerate(flags):
        if not ok:
            mismatches.append({
                "idx": i,
                "sentence": sentences[i],
                "label": labels[i],
                "pred": preds[i]
            })
    return acc, correct, total, mismatches

def main():
    records = load_jsonl(DATA_PATH)
    sentences = [r["Sentence"] for r in records]
    labels    = [r["Homograph_Phoneme_GE2PE"] for r in records]

    models = {name: GE2PE(model_path=path) for name, path in MODELS.items()}

    print(f"samples: {len(records)}\n")
    for name, m in models.items():
        acc, correct, total, mismatches = evaluate_model(m, sentences, labels, USE_RULES)
        print(f"{name:10s} | Accuracy: {acc:.2%}  ({correct}/{total})")

        if SHOW_MISMATCH_EXAMPLES and mismatches:
            # for ex in mismatches[:SHOW_MISMATCH_EXAMPLES]:
            #     print(f"   - idx={ex['idx']}:")
            #     print(f"     Sentence: {ex['sentence']}")
            #     print(f"     Label   : {ex['label']}")
            #     print(f"     Pred    : {ex['pred']}")
            # print()
            pass

if __name__ == "__main__":
    main()

