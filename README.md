# Multilabel Chest X-Ray Pathology Classification

A deep learning system for automated detection of thoracic conditions in chest radiographs, built with TensorFlow and Keras. The project covers the full ML pipeline — from raw image data through TFRecord conversion, transfer learning, ensemble construction, and GradCAM-based visual explanation.

**Final Result:** 0.780 macro AUC-ROC on a held-out test set of 1,128 images. Effusion reached 0.862 AUC, matching the performance range reported by Wang et al. (2017) despite training on a fraction of their dataset with no GPU.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Models](#models)
- [Training Strategy](#training-strategy)
- [Ensemble Methods](#ensemble-methods)
- [Results](#results)
- [GradCAM Explanations](#gradcam-explanations)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [References](#references)

---

## Project Overview

Chest radiography is one of the most common diagnostic procedures in medicine, but interpretation requires specialist expertise that is unevenly distributed — particularly in lower-resource settings. This project develops a multilabel classification system that can flag four thoracic findings from a single posterior-anterior chest X-ray:

- **Atelectasis** — partial or complete lung collapse
- **Infiltration** — diffuse parenchymal opacity
- **Effusion** — pleural fluid accumulation
- **No Finding** — normal radiograph

The system is built on transfer learning from ImageNet-pretrained backbones (DenseNet-121 and ResNet50V2), combined using two ensemble strategies, with per-class decision threshold optimisation. The entire pipeline runs on CPU — no GPU required.

---

## Problem Statement

Chest radiograph classification has three characteristics that make it harder than standard image classification:

1. **It is a multilabel problem.** A patient may have multiple concurrent findings, or none. Using a softmax multiclass head would be wrong here.

2. **The dataset is class-imbalanced.** Standard binary cross-entropy treats false negatives and false positives equally, which causes the model to converge to predicting the majority class. We address this with Asymmetric Loss (Ridnik et al., 2021), which separately controls the focus applied to hard positives and easy negatives.

3. **Predictions must be explainable.** A black-box output is not clinically usable. GradCAM is applied to confirm that the network is attending to anatomically relevant regions.

---

## Dataset

The NIH ChestX-ray14 dataset (Wang et al., 2017) contains 112,120 frontal-view chest X-rays from 30,805 patients. This project uses a pre-split subset of 7,376 images.

| Split      | Atelectasis | Infiltration | Effusion | No Finding | Total |
|------------|-------------|--------------|----------|------------|-------|
| Train      | 1,274       | 1,331        | 1,275    | 1,275      | 5,155 |
| Validation | 273         | 274          | 273      | 273        | 1,093 |
| Test       | 281         | 273          | 301      | 273        | 1,128 |

Each class accounts for roughly 25% of images in every split. The dataset is available from the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC). Images must be placed in the folder structure described in the [How to Run](#how-to-run) section.

---

## Pipeline Architecture

```
Raw PNG images (Google Drive)
        │
        ▼
Image Quality Audit (PIL — checks dimensions, pixel range, corruption)
        │
        ▼
TFRecord Conversion (8 train shards / 4 val shards / 4 test shards)
        │
        ▼
tf.data Pipeline (parallel interleave → shuffle → decode → augment → normalise → batch → prefetch)
        │
        ▼
Phase 1: Frozen Backbone Warm-Up (3 epochs, head only)
        │
        ▼
Phase 2: Full Fine-Tuning (up to 10 epochs, all layers, cosine decay)
        │
        ▼
Ensemble Construction (Weighted Average + Stacking with Logistic Regression)
        │
        ▼
Per-Class Threshold Optimisation (maximise F1 on validation set)
        │
        ▼
Evaluation + GradCAM Visualisations
```

---

## Models

### DenseNet-121

DenseNet-121 (Huang et al., 2017) uses dense connectivity — each layer receives feature maps from all preceding layers within the same dense block. For chest radiographs, this architecture is particularly effective because:

- It encourages feature reuse rather than redundancy, which helps with subtle texture-level pathology signals
- Its parameter count (~8M) makes it practical on CPU
- Wang et al. (2017) demonstrated strong performance on this exact dataset, establishing it as the standard baseline

The backbone was loaded with ImageNet weights from `tf.keras.applications.DenseNet121`. The classification layer was removed, and the 7×7×1024 feature map was pooled and passed through a custom head.

### ResNet50V2

ResNet50V2 (He et al., 2016) uses pre-activation residual units (BatchNorm and ReLU before convolution), which improves gradient flow relative to the original ResNet-50. At ~25M parameters it is larger than DenseNet-121, but its different architectural inductive bias makes it a useful second backbone for ensembling — the two models make different errors.

### Classification Head

Both backbones share the same head architecture:

```
GlobalAveragePooling2D
BatchNormalization
Dropout(0.4)
Dense(512, L2 regularisation λ=1e-4)
BatchNormalization
GELU activation
Dropout(0.2)
Dense(4, dtype=float32)   ← logits, no sigmoid here
```

---

## Training Strategy

### Loss Function: Asymmetric Loss

Standard BCE gives equal weight to easy negatives and hard positives. On imbalanced datasets this suppresses gradient signal from the minority class. The Asymmetric Loss (Ridnik et al., 2021) solves this by using separate focusing exponents for positives and negatives:

```
γ+ = 1    (light focus on hard positives)
γ- = 4    (aggressive downweighting of easy negatives)
margin m = 0.05   (discard trivially easy negatives entirely)
```

This is more principled than weighted BCE and produces better-calibrated outputs.

### Two-Phase Training Protocol

**Phase 1 — Frozen Warm-Up (3 epochs)**  
The backbone is frozen. Only the randomly initialised classification head is trained. Learning rate: cosine decay from 1e-4 → 1e-6. This prevents the large initial gradients from the random head from corrupting the pretrained backbone features.

**Phase 2 — Full Fine-Tuning (up to 10 epochs)**  
All layers are unfrozen. A 2-epoch linear warm-up brings the learning rate from 1e-5 to 1e-4, followed by cosine decay to 1e-6. The temporary AUC drop at the start of Phase 2 (0.656 → 0.623) is expected and well-documented — it resolves within two epochs as the backbone adapts.

### Optimiser and Callbacks

- **AdamW** with weight decay 1e-4 (decoupled from gradient, as in Loshchilov & Hutter, 2019)
- **ModelCheckpoint** — saves only on validation AUC improvement
- **EarlyStopping** — patience 5, restores best weights
- **ReduceLROnPlateau** — patience 3, factor 0.5

---

## Ensemble Methods

### Weighted Average Ensemble

```python
p_ensemble = w1 * p_densenet + w2 * p_resnet   # w1 + w2 = 1
```

Weights were found by evaluating 2,000 Dirichlet-sampled candidates on the validation set and selecting the combination with the highest macro AUC.

### Stacking Ensemble

The sigmoid outputs of both models (8-dimensional vector per image) were used as features for a per-class logistic regression meta-learner trained on the validation predictions. The stacking ensemble can learn that one backbone is systematically stronger for a particular class, which fixed-weight averaging cannot capture.

### Per-Class Threshold Optimisation

```python
τ* = argmax_{τ ∈ [0.1, 0.9]} F1(y_c, ŷ_c ≥ τ)
```

All optimal thresholds fell below 0.5 (range 0.38–0.45), consistent with ASL's behaviour of slightly shifting probability outputs toward the negative end.

---

## Results

### Individual Model Performance (Test Set)

| Class        | DenseNet-121 | ResNet50V2 |
|--------------|-------------|------------|
| Atelectasis  | 0.740       | 0.717      |
| Infiltration | 0.711       | 0.718      |
| Effusion     | 0.850       | 0.837      |
| No Finding   | 0.747       | 0.749      |
| **Macro AUC**| **0.762**   | **0.755**  |

### Ensemble Performance (Test Set)

| Metric         | DenseNet-121 | ResNet50V2 | Ens-WA | Ens-Stack |
|----------------|-------------|------------|--------|-----------|
| Macro AUC-ROC  | 0.762       | 0.755      | 0.777  | **0.780** |
| Macro Avg Prec | 0.531       | 0.530      | 0.559  | 0.557     |
| Macro F1       | 0.515       | 0.501      | **0.526** | 0.411  |
| Macro Precision| 0.407       | 0.358      | 0.393  | **0.641** |
| Macro Recall   | 0.730       | 0.834      | **0.801** | 0.317  |

The stacking ensemble has the highest AUC (0.780) but trades recall for precision. For a screening application where missing a disease is costlier than a false alarm, the **weighted average ensemble is the better operational choice** — higher F1 (0.526) and recall (0.801).

### Comparison with Wang et al. (2017)

| Class        | Wang et al. | This Work |
|--------------|-------------|-----------|
| Atelectasis  | 0.716       | **0.762** |
| Infiltration | 0.734       | **0.738** |
| Effusion     | 0.811       | **0.862** |
| No Finding   | 0.833       | 0.759     |
| **Macro AUC**| 0.774       | **0.780** |

Wang et al. trained on 86,524 images with GPU clusters. This comparison is contextual, not a direct equivalence claim.

---

## GradCAM Explanations

GradCAM (Selvaraju et al., 2017) was applied to the final DenseNet-121 layer to verify that predictions are driven by anatomically relevant regions:

- **Effusion** — activation concentrates in the costophrenic angle and lower hemithorax, where pleural fluid accumulates
- **Atelectasis** — activation distributed across mid and lower lung zones, consistent with lobar collapse patterns
- **No Finding** — low-intensity, diffuse activation across the full lung field (no single region drives the prediction)

This confirms the model learned genuine radiographic features rather than dataset artefacts or confounders.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/chest-xray-classification.git
cd chest-xray-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download the NIH ChestX-ray14 images from the [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) and arrange them in Google Drive as follows:

```
MyDrive/
└── sequence_images/
    ├── train/
    │   ├── Atelectasis/
    │   ├── Infiltration/
    │   ├── Effusion/
    │   └── No Finding/
    ├── validation/
    │   └── ...
    └── test/
        └── ...
```

### 4. Open the notebook

Upload `ChestXray_CPU_Professional_1.ipynb` to [Google Colab](https://colab.research.google.com). Mount your Google Drive and update the `DATASET_DIR` path in Section 2 to match your folder location.

The notebook handles everything else: TFRecord conversion, training, ensemble construction, evaluation, and GradCAM visualisation.

### 5. Runtime note

All training was done on a **standard Colab CPU runtime** (Intel Xeon, 2 vCPUs, 12 GB RAM). No GPU needed. Total training time for both models is approximately 7 hours. TFRecord conversion runs once and is cached.

---

## Requirements

```
tensorflow==2.19.0
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
tqdm
```

See `requirements.txt` for pinned versions.

---

## Repository Structure

```
chest-xray-classification/
├── ChestXray_CPU_Professional_1.ipynb   # Full training and evaluation notebook
├── requirements.txt
├── README.md
└── report/
    └── x_ray_images_report.pdf          # Full project report (BSc thesis)
```

Model weights are not included in this repository due to file size. The notebook will train them from scratch, or you can load your own checkpoint by modifying the path in the ensemble construction section.

---

## References

1. Wang, X. et al. (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks. *CVPR*.
2. Huang, G. et al. (2017). Densely connected convolutional networks. *CVPR*.
3. He, K. et al. (2016). Identity mappings in deep residual networks. *ECCV*.
4. Ridnik, T. et al. (2021). Asymmetric loss for multi-label classification. *ICCV*.
5. Selvaraju, R.R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
6. Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
7. Tajbakhsh, N. et al. (2016). Convolutional neural networks for medical image analysis: Full training or fine tuning? *IEEE Transactions on Medical Imaging*.

---

## Acknowledgements

Dataset provided by the NIH Clinical Center under an open research license. Training conducted on Google Colaboratory. Project completed as a Bachelor of Science final year project, Department of Data Science, The Islamia University of Bahawalpur (2021–2025).
