# Associate Researcher Assignment

This project is a minimal text classification pipeline using Hugging Face 
Transformers.  

It uses the dataset training-dataset.csv with binary labels for heavy 
menstrual bleeding.

------------------------------------------------------------

Repo Structure
associate_researcher_assignment/
├── data/                   # dataset(s)
│   └── training-dataset.csv
├── src/                    # training/eval code
│   └── train.py
├── notebooks/              # jupyter notebooks
├── run.sh                  # bash launcher
├── requirements.txt        # required python libraries
└── README.md

------------------------------------------------------------

Setup
(using venv)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

(or with conda)
conda create -n ar_env python=3.10 -y
conda activate ar_env
pip install -r requirements.txt

------------------------------------------------------------

Run
python src/train.py

chmod +x run.sh
./run.sh

------------------------------------------------------------

Output
- 5-fold CV with per-fold validation accuracy, F1, and a confusion matrix
- When using run.sh, all logs are saved to train.log







