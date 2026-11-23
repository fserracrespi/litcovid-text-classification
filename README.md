# Biomedical Text Classification with PyTorch (LitCovid)

Lab Exercise 1 for **Deep Learning for NLP**: **multi-label** classification of COVID-19 scientific articles into 7 research topics, using two neural networks implemented from scratch in PyTorch.

## Task

Assign each article (title + abstract + keywords) one or more of the following topics:

`Treatment` · `Diagnosis` · `Prevention` · `Mechanism` · `Transmission` · `Epidemic Forecasting` · `Case Report`

**Dataset:** [LitCovid – BioCreative VII](https://huggingface.co/datasets/KushT/LitCovid_BioCreative) (24,960 training articles and 6,239 development articles).

> In the Hugging Face version, the 6,239-article development set is stored under the `test` split. The notebook detects this automatically.

## Models

| Model | Description |
|---|---|
| **CBOW / Bag-of-Words** | Embeddings → mean pooling (ignoring `<PAD>`) → ReLU hidden layer → 7 logits |
| **CNN** | Embeddings → convolutions with window sizes 2, 3 and 5 → ReLU → max pooling → 7 logits |

- Loss: `BCEWithLogitsLoss` (with optional `pos_weight` to handle class imbalance)
- Optimizer: Adam (`lr = 1e-3`)
- Metrics: precision, recall, micro-F1 and macro-F1

## Notebook contents

1. Dataset exploration and label distribution
2. Preprocessing: cleaning, vocabulary, conversion to ids, padding/truncation
3. Tensors and DataLoaders
4. BOW and CNN model implementations
5. Loss function
6. Training loop with validation
7. Evaluation (thresholds 0.5 and 0.3)
8. Experiments and architecture comparison
9. Discussion
10. Extensions: `pos_weight`, per-label thresholds, pretrained embeddings, transformers

## How to run

```bash
git clone https://github.com/<username>/litcovid-text-classification.git
cd litcovid-text-classification
pip install -r requirements.txt
jupyter notebook assignment1_litcovid.ipynb
```

The notebook can also be opened in **Google Colab** (a GPU is recommended for the CNN).

Main parameters (first cell of the notebook):

| Parameter | Value | Description |
|---|---|---|
| `MAX_LEN` | 500 | Maximum sequence length |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `EPOCHS` | 5 | Epochs per experiment |
| `QUICK_RUN` | `False` | `True` = use a small subset to test the code quickly |

The random seed is fixed (`SEED = 42`) so results are reproducible.

## Requirements

```
torch
datasets
scikit-learn
numpy
pandas
matplotlib
jupyter
```

## Author

<Francesc SERRA CRESPÍ> — <University Paris-Saclay / M2 IA>, <2026-2027>
