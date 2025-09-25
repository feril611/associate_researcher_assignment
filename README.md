# Associate Researcher Assignment

Minimal text classification pipeline using Hugging Face Transformers on 
`training-dataset.csv` (binary labels).  
Includes a one-command runner (`run.sh`) that creates/uses a local virtual 
environment and logs to `train.log`.

---

## Repo Structure
associate_researcher_assignment/
├── data/
│   └── training-dataset.csv
├── src/
│   └── train.py
├── notebooks/
├── run.sh
├── requirements.txt
└── README.md

---

## Quick Start 
```bash
# from repo root
chmod +x run.sh
./run.sh           


---

## What `run.sh` does
1. Creates/activates a local Python virtual environment at `.venv/`.
2. Installs packages from `requirements.txt` (first run).
3. Runs `python -u src/train.py` and writes all output to a log file 
(default `train.log`).

> The training script resolves the dataset path relative to the repo root:
> `data/training-dataset.csv`

---

## Manual Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

---

## Outputs
- Per-fold validation **accuracy**, **F1**, and a **confusion matrix** in 
the console/log.
- Log saved to `train.log`






