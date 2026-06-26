# Multimodal Alzheimer's Disease Detection using a KNN Clinical Bridge Meta-Learner

A multimodal machine learning framework for Alzheimer's Disease (AD) detection by integrating **MRI** and **speech** modalities across two independent datasets with **zero overlapping subjects**.

**Author:** Ratanjot Singh  
**Registration No.:** 22BBS0058  
**Department:** School of Computer Science and Engineering (SCOPE)  
**Institution:** VIT Vellore

---

# Overview

Alzheimer's Disease diagnosis increasingly benefits from multimodal data such as structural MRI and spontaneous speech. However, publicly available datasets containing both modalities for the same individuals are extremely limited.

This project proposes a **cross-dataset multimodal fusion framework** using:

- **OASIS-1** for MRI
- **ADReSS / DementiaBank** for speech

Since these datasets contain **no common subjects**, conventional multimodal fusion cannot be applied.

To address this challenge, a **KNN Clinical Bridge Meta-Learner** is proposed, where shared clinical variables (**Age** and **MMSE**) are used to estimate the missing modality probability through inverse-distance weighted K-Nearest Neighbours (KNN). These probabilities are then combined using a calibrated meta-learner.

---

# Pipeline

```
                         OASIS Dataset
                    ┌────────────────────┐
                    │   brain.mgz MRI    │
                    │   FreeSurfer Stats │
                    └──────────┬─────────┘
                               │
                  MRI Feature Extraction
                               │
                               ▼
                        XGBoost Classifier
                               │
                               ▼
                             P_mri
                                \
                                 \
                                  \
                                   ▼
                      KNN Clinical Bridge
                     (Age + MMSE Similarity)
                                   ▲
                                  /
                                 /
                                /
                            P_audio
                               ▲
                      XGBoost Classifier
                               │
          Acoustic + Transcript Feature Extraction
                               │
                    ┌──────────┴──────────┐
                    │  ADReSS Dataset     │
                    │ Speech + Transcript │
                    └─────────────────────┘

                               ▼
                  Calibrated Meta-Learner

                               ▼
                 Alzheimer's Disease Prediction
```

---

# Methodology

The framework consists of three stages.

## Stage 1 — MRI Pipeline (OASIS)

### Input

- Brain MRI (`brain.mgz`)
- FreeSurfer morphometric statistics

### Feature Extraction

### Brain MRI

- 3D Convolutional Neural Network
- 65,536-dimensional embedding
- StandardScaler
- Incremental PCA
- 120 principal components

### FreeSurfer

Extracted cortical and subcortical morphometric measurements including:

- Cortical thickness
- Surface area
- Subcortical volumes

Processing:

- Median imputation
- StandardScaler
- PCA (95% explained variance)

Final representation:

```
Brain.mgz embedding
        +
FreeSurfer features
```

### MRI Classification

Classifier:

- XGBoost

Output:

```
P_mri
```

---

## Stage 2 — Audio Pipeline (ADReSS)

### Input

- Speech recordings
- CHAT transcripts

### Acoustic Features

#### openSMILE ComParE 2016

- 6373-dimensional acoustic descriptor

Processing:

- StandardScaler
- PCA (95%)

---

#### librosa Features

Includes:

- MFCC
- Chroma
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- RMS energy
- Zero crossing rate

Processing:

- StandardScaler
- PCA

---

### Text Features

TF-IDF representation

Processing:

- Unigrams + Bigrams
- PCA

---

### Linguistic Features

Extracted from CHAT transcripts:

- Word count
- Sentence count
- Type-token ratio
- Vocabulary size
- Fillers
- Pronoun count
- Information units
- Verbosity
- Hapax ratio

Processing:

- StandardScaler

---

### Audio Classification

Features combined:

```
openSMILE
+
librosa
+
TF-IDF
+
Linguistic
```

Classifier:

- XGBoost

Output:

```
P_audio
```

---

## Stage 3 — KNN Clinical Bridge

Because MRI subjects have no speech recordings and speech subjects have no MRI scans, missing modality probabilities must be estimated.

Shared clinical variables:

- Age
- MMSE

Procedure:

1. Standardize Age and MMSE.
2. Find the **K nearest neighbours** (K = 9).
3. Compute inverse-distance weighted probability.
4. Estimate the missing modality probability.

Additional reliability measures:

- Neighbour entropy
- Mean neighbour distance

These are provided to a calibrated Logistic Regression meta-learner.

Output:

```
P_final
```

---

# Model Architecture

## MRI Model

```
brain.mgz
      │
      ▼
   3D CNN
      │
Incremental PCA
      │
      ├────────────┐
      │            │
FreeSurfer      PCA
      │            │
      └──────┬─────┘
             ▼
         XGBoost
             ▼
           P_mri
```

---

## Audio Model

```
Speech
   │
   ├───────────┐
   │           │
openSMILE   librosa
   │           │
 PCA         PCA
   │           │
   └────┬──────┘
        │

Transcript
     │
 TF-IDF
     │
    PCA

Linguistic Features
     │
 StandardScaler

       │
       ▼

Concatenate

       ▼

XGBoost

       ▼

P_audio
```

---

## Fusion Model

```
P_mri

Age

MMSE
        │

        ▼

KNN

        ▼

Estimated P_audio

        ▼

Entropy

Distance

        ▼

Meta Learner

        ▼

Final Prediction
```

---

# Experimental Results

Evaluation performed using **5-Fold Stratified Out-of-Fold (OOF) Cross Validation**.

| Pipeline | AUC | Accuracy | Subjects |
|-----------|-----|----------|----------|
| MRI Pipeline | **0.9710** | **88.94%** | 425 |
| Audio Pipeline | **0.8920** | **84.26%** | 108 |
| KNN Meta-Learner | **0.9425** | **90.24%** | 533 |

---

# Key Contributions

- Cross-dataset MRI and speech fusion with **zero overlapping subjects**
- Novel **KNN Clinical Bridge** for missing modality estimation
- Reliability-aware fusion using:
  - neighbour entropy
  - neighbour distance
- Zero data leakage through fold-wise preprocessing
- Out-of-Fold probability generation
- Calibrated meta-learning using isotonic calibration

---

# Repository Structure

```text
alzheimer-knn-bridge/
│
├── AlzheimerDetection_Full.ipynb      # Complete training pipeline + interactive demo
├── config_example.py                  # Example dataset paths
├── requirements.txt                   # Python dependencies
├── README.md
└── .gitignore
```

---

# Datasets

The datasets are **not distributed** with this repository because of licensing restrictions.

## OASIS-1

Structural MRI dataset.

Download:

https://www.oasis-brains.org

After downloading, update the dataset path in `config.py`.

---

## ADReSS / DementiaBank

Speech and transcript dataset.

Download:

https://dementia.talkbank.org

After downloading, update the dataset path in `config.py`.

---

# Installation

Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/alzheimer-knn-bridge.git

cd alzheimer-knn-bridge
```

Install the required Python packages.

```bash
pip install -r requirements.txt
```

Install **FreeSurfer** separately for MRI preprocessing.

Official installation guide:

https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall

---

# Configuration

Copy:

```text
config_example.py
```

Rename it to:

```text
config.py
```

Update the dataset locations according to your system.

---

# Usage

Open:

```text
AlzheimerDetection_Full.ipynb
```

Run all notebook cells sequentially.

The notebook performs:

- MRI preprocessing
- MRI model training
- Audio preprocessing
- Audio model training
- Out-of-Fold probability generation
- KNN Clinical Bridge construction
- Meta-learner training
- Performance evaluation

---

# Interactive Demo

The final section of **AlzheimerDetection_Full.ipynb** contains an interactive demonstration for predicting Alzheimer's Disease on a new subject after all models have been trained.

No separate demo application is required.

---

# Technologies Used

- Python
- PyTorch
- XGBoost
- scikit-learn
- FreeSurfer
- nibabel
- openSMILE
- librosa
- NumPy
- pandas

---

# Citation

If you use this work, please cite:

```text
Singh, R. (2026).

Cross-Dataset Multimodal Fusion for Alzheimer's Disease Detection
Using a KNN Clinical Bridge Meta-Learner.

B.Tech Capstone Project,
Department of Analytics,
VIT Vellore.
```

---

# License

**Code**

MIT License

**Datasets**

The OASIS and DementiaBank datasets are subject to their respective data usage agreements and are **not included** in this repository.

---

# Acknowledgements

- VIT Vellore
- School of Computer Science and Engineering (SCOPE)
- OASIS Research Group
- DementiaBank / TalkBank
- FreeSurfer Development Team
- openSMILE Developers
- librosa Development Team
```
