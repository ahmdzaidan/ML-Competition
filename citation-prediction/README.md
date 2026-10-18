# Citation Link Prediction in Academic Paper Networks

SPECTER2 & Deep Neural Network approach to predict citation relationships 
between academic papers, developed for Data Science Competition - Gammafest IPB 2025

---

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Competition](https://img.shields.io/badge/Competition-Gammafest%20IPB%202025-blueviolet?style=flat)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Problem Formulation & Dataset

This dataset consists of pairs of academic papers, each accompanied by bibliographic metadata such as title, abstract, and publication year. Each record represents a potential citation link where the first document (`paper`) may cite the second document (`referenced_paper`).

The objective is to build a machine learning model to predict the value of **`is_referenced`**, determining whether the citation link actually exists (`1`) or not (`0`), based on the available information.

### Dataset Files
- **Paper Database Folder (`papers/`):** Contains full-text documents in `.txt` format, with filenames corresponding to `paper_id`.
- **`papers_metadata.csv`:** Comprehensive metadata for each paper.
- **`train.csv`:** Document pairs with ground truth labels indicating whether `paper` cites `referenced_paper`.
- **`test.csv`:** Unlabeled document pairs to be predicted.
- **`sample_submission.csv`:** Sample submission format for test data predictions.

### Document Metadata Features (`papers_metadata.csv`)
| Column | Description |
| :--- | :--- |
| `paper_id` | Unique document identifier |
| `doi` | Digital Object Identifier (if available) |
| `title` | Paper title |
| `publication_year` | Year of publication |
| `publication_date` | Date of publication |
| `cited_by_count` | Number of times this document has been cited by other papers |
| `type` | Document category (e.g., *article*, *conference*, etc.) |
| `authors` | List of authors |
| `concepts` | Topics or academic field concepts |

---

## 🏗️ Methodology & Architecture

Our solution combines scientific document semantic representations from a domain-specialized transformer (**SPECTER2**) with temporal bibliographic feature engineering and a regularized **Deep Neural Network**.

```
       [ Paper A Full-Text ]                   [ Paper B Full-Text ]
                 │                                       │
                 ▼                                       ▼
       ┌───────────────────┐                   ┌───────────────────┐
       │  SPECTER2 [CLS]   │                   │  SPECTER2 [CLS]   │
       │    (768 dims)     │                   │    (768 dims)     │
       └─────────┬─────────┘                   └─────────┬─────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ Pairwise Semantic Interaction │
                     │   [ u , v , |u - v| ]         │  (2,304 dims)
                     └───────────────┬───────────────┘
                                     │
     [ Bibliographic Metadata ]      │
       • ΔYear (Year_A - Year_B)     │
       • Future flag (ΔYear < 0)     │
       • Same paper type (0/1)       ├─────────────────────────────────┐
       • Citation count (Paper A)    │                                 │
       • Citation count (Paper B)    │                                 │
     (5 engineered features)         │                                 │
                 │                   ▼                                 ▼
                 └───────────► [ Feature Fusion ] ──────────► [ StandardScaler ]
                                 (2,309 dims)                          │
                                                                       ▼
                                                          ┌───────────────────────────┐
                                                          │   Deep Neural Network     │
                                                          │  Dense 768 (BN + LeakyReLU)│
                                                          │  Dropout(0.3)             │
                                                          │  Dense 512 (BN + LeakyReLU)│
                                                          │  Dropout(0.3)             │
                                                          │  Dense 256 (BN + LeakyReLU)│
                                                          │  Dropout(0.2)             │
                                                          │  Dense 128 (BN + LeakyReLU)│
                                                          │  Dense 1 (Sigmoid)        │
                                                          └─────────────┬─────────────┘
                                                                        │
                                                                        ▼
                                                          ┌───────────────────────────┐
                                                          │   Threshold Tuning (MCC)  │
                                                          │   Optimal Threshold: 0.35 │
                                                          └─────────────┬─────────────┘
                                                                        │
                                                                        ▼
                                                             Prediction: 0 or 1
```

### 1. Document Semantic Embeddings (SPECTER2)
- Utilizing pre-trained **`allenai/specter2_base`**, specifically trained on scientific citation graphs via contrastive learning.
- The interaction vector for a paper pair $(u, v)$ is constructed via concatenation and element-wise absolute difference:
  $$\mathbf{x}_{\text{text}} = [u \,\|\, v \,\|\, |u - v|] \in \mathbb{R}^{2304}$$

### 2. Bibliographic Feature Engineering
- **$\Delta\text{Year}$:** Year difference ($\text{Year}_A - \text{Year}_B$).
- **`future`:** Binary flag indicating whether the cited paper was published after the citing paper.
- **`same_type`:** Indicator for identical paper publication category ($1$ if identical, $0$ otherwise).
- **`cited_p` & `cited_r`:** Historical citation in-degree counts for citing and cited papers.

### 3. Classification Head (Deep Neural Network)
- A 4-hidden-layer MLP ($768 \to 512 \to 256 \to 128 \to 1$) with **Batch Normalization**, **LeakyReLU** activations ($\alpha=0.01$), and **Dropout** regularization (0.3 / 0.2) to prevent overfitting on the dense embedding space.
- Optimizer: Adam ($10^{-4}$), binary cross-entropy loss, and `ReduceLROnPlateau` scheduling.

### 4. Decision Threshold Optimization (Matthews Correlation Coefficient)
- Due to the extreme class imbalance (~1.04% positive citations), an exhaustive grid search was conducted over $[0.00, 1.00]$ to maximize the **Matthews Correlation Coefficient (MCC)** on the stratified validation split.
- **Optimal Decision Threshold:** `0.35`
- **Validation MCC:** `0.5306`

---

## 📊 Experimental Results

Evaluated on the stratified validation set (82,139 candidate paper pairs: 81,281 negatives, 858 positives):

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Optimal MCC** | **0.5306** | Peak achieved at decision threshold $t^* = 0.35$ |
| **Accuracy** | **99.0%** | |
| **Precision (Class 1 - Citation)** | **0.55** | |
| **Recall (Class 1 - Citation)** | **0.52** | High sensitivity despite $<1.1\%$ positive base rate |
| **F1-Score (Class 1)** | **0.54** | Balanced trade-off for citation link discovery |
| **Precision / Recall (Class 0)** | **0.99 / 1.00** | Robust rejection of non-citations |

---

## 📁 Repository Structure

```
citation-prediction/
├── data/
│   └── README.md                # Dataset layout instructions and metadata schema
├── models/
│   ├── README.md                # Pre-trained model weights & scaler documentation
│   ├── scaler.pkl               # Fitted StandardScaler object
│   └── specter2_nn_model.h5     # Trained Keras Deep Neural Network weights
├── notebooks/
│   ├── DSC.ipynb                # Main research notebook (EDA, training & evaluation)
│   └── inference.ipynb          # Standalone inference notebook using saved artifacts
├── .gitattributes               # Git LFS configuration for binary weights
├── .gitignore                   # Excludes raw datasets, notebook checkpoints, and venv
├── LICENSE                      # MIT License
├── requirements.txt             # Environment dependencies
└── README.md                    # Project documentation
```

---

## 💻 Running the Notebooks

This project is entirely **Jupyter Notebook-centric (`.ipynb`)**:

### 1. Environment Setup
Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Full Experimentation & Training (`notebooks/DSC.ipynb`)
Open `notebooks/DSC.ipynb` in Google Colab (with GPU accelerator) or local Jupyter:
- Covers the full end-to-end pipeline: data preprocessing, SPECTER2 text embedding extraction, feature engineering, Keras neural network training, MCC threshold search, and final submission generation.

### 3. Quick Inference on New Test Data (`notebooks/inference.ipynb`)
To generate predictions without retraining:
- Open and run `notebooks/inference.ipynb`.
- It loads `models/scaler.pkl` and `models/specter2_nn_model.h5` directly to predict on `data/test.csv`.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 🤝 Acknowledgements
- **Gammafest IPB 2025** Organizers & Data Science Competition Committee.
- **Allen Institute for AI (AI2)** for the [SPECTER2](https://huggingface.co/allenai/specter2_base) foundation model.
