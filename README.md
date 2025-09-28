# ParsHomo: Persian Homograph Disambiguation with T5

This repository accompanies the paper  
**"ParsHomo: A T5-Powered Approach to High-Precision Persian Homograph Disambiguation"**.  

It contains benchmark data and scripts for training, evaluation, and inference with fine-tuned T5-based architectures (ByT5 and mT5).  
⚠️ **Note:** The main large-scale datasets and pretrained models are **not included** due to size and licensing restrictions.  

---

## 📂 Repository Structure

```
ParsHomo
├── ByT5
│ ├── Homograph_ByT5_Train.py
│ ├── Homograph_ByT5_Evaluation.py
│ └── Homograph_ByT5_Inference.py
│
├── mT5
│ ├── Homograph_mT5_Train.py
│ ├── Homograph_mT5_Evaluation.py
│ └── Homograph_mT5_Inference.py
│
├── Benchmark_Dataset.jsonl # Sample benchmark dataset
├── benchamrk_Ge2pe.py # Baseline GE2PE benchmarking
├── LICENSE
└── .gitignore
```
---

## 🚀 Usage

### 1. Install Requirements
```bash
pip install requirement.txt
```


### 2. Training

Run training for ByT5 or mT5:
```bash
python Homograph_ByT5_Train.py
python Homograph_mT5_Train.py
```

### 3. Evaluation

Evaluate trained models on the benchmark dataset:
```bash
python Homograph_ByT5_Evaluation.py
python Homograph_mT5_Evaluation.py
```


### 4. Inference

Test homograph disambiguation with a custom input:

```bash
python Homograph_ByT5_Inference.py
python Homograph_mT5_Inference.py
```
## 📊 Benchmark Dataset

We provide a small benchmark dataset (Benchmark_Dataset.jsonl) with annotated sentences for evaluation and reproducibility.

Each record in `Benchmark_Dataset.jsonl` looks like this:

```json
{
  "Sentence": "همه‌ی عالم را نوری از روشنایی خدا بدان",
  "Homograph": "عالم",
  "Homograph_Phoneme_Homorich": "?Alam",
  "Homograph_Phoneme_eSpeak": "?Alam",
  "Homograph_Phoneme_GE2PE": "@al/m"
}
```
## 📖 Citation

If you use this repository in your research, please cite our paper:
```article
@inproceedings{jalali2025parshomo,
  title     = {ParsHomo: A T5-Powered Approach to High-Precision Persian Homograph Disambiguation},
  author    = {Hasan Jalali and Taha Mohaddesi},
  booktitle = {Proceedings of the 15th International Conference on Computer and Knowledge Engineering (ICCKE)},
  year      = {2025},
  publisher = {IEEE},
}
```
## 📬 Contact
For any inquiries or contributions, please contact hasanjalali@ut.ac.ir, mohaddesi.t@gmail.com.
