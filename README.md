# Deep Learning for NLP – Assignment 1

This repository contains the code for **Assignment 1** of the course **<DL for NLP>** (University Paris-Saclay/ M2 AI, 2026-2027).

**Topic:** Biomedical Text Classification with PyTorch

## Description

The goal of this assignment is to automatically assign COVID-19 scientific articles to one or more research topics (**multi-label classification**) using neural networks implemented from scratch in PyTorch.

The seven topics are:

`Treatment` · `Diagnosis` · `Prevention` · `Mechanism` · `Transmission` · `Epidemic Forecasting` · `Case Report`

Two architectures are implemented and compared:

1. A **Bag-of-Words (BOW)** classifier based on word embeddings.
2. A **Convolutional Neural Network (CNN)** for text classification.

## Dataset

[LitCovid – BioCreative VII](https://huggingface.co/datasets/KushT/LitCovid_BioCreative), loaded with the Hugging Face `datasets` library. The training and development sets are used.

## Repository structure

```
.
├── assignment1.ipynb   # Notebook with the full assignment
├── requirements.txt    # Python dependencies
└── README.md
```

## Notebook contents

1. Dataset exploration and preprocessing
2. Vocabulary construction and numerical representation
3. Implementation of the BOW classifier
4. Implementation of the CNN classifier
5. Training and validation procedures
6. Evaluation (precision, recall, micro-F1, macro-F1)
7. Comparison of the two architectures
8. Answers to the discussion questions

## How to run

```bash
git clone https://github.com/fserracrespi/litcovid-text-classification.git
cd litcovid-text-classification
pip install -r requirements.txt
jupyter notebook assignment1.ipynb
```

The notebook can also be run on Google Colab (a GPU is recommended).

## Author

**<Your name>**
