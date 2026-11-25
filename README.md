# PRISM-Net: Physics-Informed Radiomic-Neural Cross-Attention Network for Leakage-Free Renal CT Phenotyping

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-ConvNeXtV2%20%2B%20Swin-blue.svg)](https://github.com/huggingface/pytorch-image-models)
[![Validation](https://img.shields.io/badge/Protocol-5--Fold%20StratifiedGroupKFold-2b5c8f.svg)]()
[![Calibration](https://img.shields.io/badge/Uncertainty-Split%20Conformal%20Prediction-3b9ab2.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**PRISM-Net** (**P**hysics-Informed **R**adiomic **I**ntegrated **S**patial **M**ulti-Stream **Net**work) is a leakage-audited, uncertainty-calibrated deep learning framework for multi-class renal pathology triage (`Normal`, `Cyst`, `Tumor`, and `Stone`) on non-contrast/contrast-enhanced abdominal CT slices.

Unlike conventional 2D CT classifiers that suffer from severe **volumetric slice-level data leakage** (artificially inflating accuracy to $100.00\%$), PRISM-Net enforces strict **pseudo-patient clustering** (`StratifiedGroupKFold` across 427 reconstructed patient studies), couples a **3-channel Pseudo-Hounsfield Unit (Pseudo-HU) physics stem** with a dual-stream **ConvNeXt-V2-Tiny + Swin-Tiny** backbone, extracts **12 differentiable $56 \times 56$ spatial radiomic descriptors** end-to-end, fuses multi-scale tokens via **Radiomic-Neural Cross-Attention (RNCA)**, and provides finite-sample clinical coverage guarantees via **Split Conformal Prediction**.

---


<img width="1600" height="838" alt="fig1_leakage_audit" src="https://github.com/user-attachments/assets/50ec78ae-7cf1-4073-acca-5663ea3508bd" />
<img width="3873" height="2676" alt="fig1_raw_eda_statistics" src="https://github.com/user-attachments/assets/dcc1ab8d-eee2-47f1-b760-b24ebe645151" />
<img width="3270" height="3348" alt="fig2_raw_ct_samples" src="https://github.com/user-attachments/assets/9ec0de9e-6617-4a24-a6b5-3c4ece8dc9c7" />
<img width="2826" height="1291" alt="fig3_ablation_trajectory" src="https://github.com/user-attachments/assets/2eee8a1b-fc9e-4fdf-ae9d-c61185cd19b1" />
<img width="3721" height="1324" alt="fig2_sota_benchmark" src="https://github.com/user-attachments/assets/6bffb622-66e4-493d-8a30-b82d2a998759" />
<img width="3570" height="1354" alt="fig6_roc_pr_curves" src="https://github.com/user-attachments/assets/aae31e31-e5ee-4622-b340-b0bb90ccc3e0" />
<img width="4753" height="1414" alt="fig4_prism_evaluation_dashboard" src="https://github.com/user-attachments/assets/c2de9bd5-9835-47b8-a7f0-b910607363dd" />
<img width="3466" height="1355" alt="fig4_calibration_confusion" src="https://github.com/user-attachments/assets/16b6bee4-76d1-4416-b850-1cc1f73edcce" />
<img width="3247" height="3348" alt="fig3_pseudo_hu_decomposition" src="https://github.com/user-attachments/assets/30757fcb-5825-4a10-aa91-67ba58229f41" />

<img width="4583" height="3497" alt="fig9_full_pipeline_walkthrough" src="https://github.com/user-attachments/assets/e5ad47a4-e9d9-419a-a4b7-aea0d2915510" />
<img width="4395" height="3711" alt="fig8_raw_vs_predicted_4x5" src="https://github.com/user-attachments/assets/4023ce18-7fc5-4ff7-abd3-d7ee3d1e71b9" />
<img width="3536" height="2742" alt="fig7_qualitative_radiomic_saliency" src="https://github.com/user-attachments/assets/038b401c-7c00-4ad2-a42f-275168653702" />
<img width="2223" height="1835" alt="fig13_tsne_penultimate_manifold" src="https://github.com/user-attachments/assets/6f869e70-b3ee-4585-a414-3db1117baf9c" />
<img width="3575" height="1324" alt="fig12_conformal_risk_efficiency" src="https://github.com/user-attachments/assets/422948d0-14bd-4622-a21f-695f501a1311" />
<img width="4019" height="2287" alt="fig11_radiomic_feature_distributions" src="https://github.com/user-attachments/assets/8cee6c6d-2c4c-421f-be15-fb5c01dfbef1" />
<img width="4395" height="3711" alt="fig10_roi_contours_4x5" src="https://github.com/user-attachments/assets/30b71352-634d-448c-aa56-f731c66fec0b" />







## Key Highlights & Contributions

* **Volumetric Slice Data Leakage Audit:** Demonstrates across four architectures (`ResNet50`, `ConvNeXt-V2-Tiny`, `Swin-Tiny`, and `PRISM-Net`) that standard random slice splitting leaks adjacent slices of the same patient across train/validation folds—inflating `ResNet50` accuracy from **$91.05\%$ to $100.00\%$ ($+8.95\%$ inflation)** and MCC from **$0.8717$ to $1.0000$ ($+0.1283$ inflation)**.
* **Physics-Informed Pseudo-HU Decomposition:** Transforms single-channel grayscale CT intensities into three window-level attenuation representations (Soft-Tissue, Corticomedullary/Nephrographic Contrast, and Bone/Calculus) prior to dual-stream feature encoding.
* **Blind Acquisition FiLM Conditioning:** Modulates local and global feature maps using Feature-wise Linear Modulation ($\text{FiLM}$) driven by slice-level geometric and intensity acquisition descriptors without requiring DICOM metadata headers.
* **Differentiable $56 \times 56$ Radiomics & Cross-Attention (RNCA):** Computes 12 closed-form, differentiable statistical and directional GLCM texture descriptors weighted by a learned $56 \times 56$ spatial attention map, querying multi-scale visual tokens via multi-head cross-attention.
* **Calibrated Clinical Safety Guarantees:** Achieves **$96.63\%$ accuracy**, **$0.9518$ MCC**, **$0.9968$ Macro-AUROC**, **$93.91\%$ Malignant Tumor Sensitivity at $99\%$ Specificity**, and **$0.0322$ ECE** across all $11,929$ out-of-fold slices, with **Split Conformal Prediction** delivering **$95.09\%$ empirical coverage** ($97.15\%$ singleton predictions at $97.49\%$ singleton accuracy) at target $\alpha = 0.05$.

---

## Mathematical Formulation

### 1. Physics-Informed 3-Channel Pseudo-HU Stem
Given a normalized single-channel CT slice $I \in [0, 1]^{H \times W}$, PRISM-Net constructs a 3-channel radiological window tensor $\mathbf{X}_{\text{HU}} \in [0, 1]^{3 \times H \times W}$ targeting renal parenchyma, vascular/mass enhancement, and high-attenuation calculi:

$$\mathbf{X}_{\text{HU}}^{(c)}(u, v) = \text{clip}\!\left(\frac{I(u, v) - \mu_c}{\sigma_c} + 0.5,\; 0,\; 1\right), \quad c \in \{\text{Soft}, \text{Mass}, \text{Stone}\}$$

### 2. Blind Acquisition FiLM Conditioning
From raw slice geometry and intensity quantiles, a 4-dimensional acquisition descriptor $\mathbf{a} \in \mathbb{R}^4$ is extracted and mapped to channel-wise affine modulation parameters $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ initialized at identity:

$$\text{FiLM}(\mathbf{F}, \mathbf{a}) = \left(\mathbf{1} + \boldsymbol{\gamma}(\mathbf{a})\right) \odot \mathbf{F} + \boldsymbol{\beta}(\mathbf{a}), \quad [\boldsymbol{\gamma}(\mathbf{a}), \boldsymbol{\beta}(\mathbf{a})] = W_2\,\text{GELU}(W_1 \mathbf{a})$$

### 3. Differentiable $56 \times 56$ Radiomic Feature Extractor
From the local `ConvNeXt-V2-Tiny` feature map $\mathbf{F}_{\text{loc}} \in \mathbb{R}^{C_l \times 56 \times 56}$, a spatial attention head predicts logits $\mathbf{A} \in \mathbb{R}^{1 \times 56 \times 56}$, normalized into a probability mass distribution $\mathbf{W} = \text{softmax}(\text{vec}(\mathbf{A})) \in \Delta^{56 \times 56}$. Downsampling the soft-tissue channel to $\tilde{I} \in \mathbb{R}^{56 \times 56}$, PRISM-Net extracts 12 differentiable radiomic descriptors $\mathbf{r} \in \mathbb{R}^{12}$:

* **First- to Fourth-Order Weighted Moments & Information:**
  $$\mu_w = \sum_{i,j} W_{i,j}\tilde{I}_{i,j}, \quad \sigma_w^2 = \sum_{i,j} W_{i,j}(\tilde{I}_{i,j} - \mu_w)^2, \quad \gamma_1 = \sum_{i,j} W_{i,j}\!\left(\frac{\tilde{I}_{i,j}-\mu_w}{\sqrt{\sigma_w^2 + \epsilon}}\right)^3$$
  $$\gamma_2 = \sum_{i,j} W_{i,j}\!\left(\frac{\tilde{I}_{i,j}-\mu_w}{\sqrt{\sigma_w^2 + \epsilon}}\right)^4, \quad E_w = \sum_{i,j} W_{i,j}\tilde{I}_{i,j}^2, \quad H_w = -\sum_{i,j} W_{i,j}\log(W_{i,j} + \epsilon)$$

* **Directional Spatial Contrast & Inverse Difference Moment (Homogeneity) across Horizontal ($h$), Vertical ($v$), and Diagonal ($d$) offsets $\Delta_{\delta}\tilde{I}$:**
  $$\text{Contrast}_{\delta} = \sum_{i,j} W_{i,j}^{(\delta)} \left(\Delta_{\delta}\tilde{I}_{i,j}\right)^2, \quad \text{Homogeneity}_{\delta} = \sum_{i,j} \frac{W_{i,j}^{(\delta)}}{1 + \left\vert{}\Delta_{\delta}\tilde{I}_{i,j}\right\vert{}}, \quad \delta \in \{h, v, d\}$$

### 4. Radiomic-Neural Cross-Attention (RNCA) & Dual-Stream Fusion
The radiomic vector $\mathbf{r} \in \mathbb{R}^{12}$ is projected into a query token $\mathbf{q} = \phi_q(\mathbf{r}) \in \mathbb{R}^{1 \times d}$ ($d = 384$) that attends over concatenated local (`ConvNeXt-V2-Tiny` pooled to $7 \times 7 = 49$ tokens) and global (`Swin-Tiny` $7 \times 7 = 49$ tokens) visual representations $\mathbf{V} \in \mathbb{R}^{98 \times d}$:

$$\mathbf{z}_{\text{RNCA}} = \text{LayerNorm}\!\left(\mathbf{q} + \text{MHA}(\mathbf{q}W_Q,\; \mathbf{V}W_K,\; \mathbf{V}W_V)\right) + \mathbf{g}_{\text{global}}$$

The final representation $\mathbf{z} = [\text{GAP}(\mathbf{F}_{\text{loc}}) \,\Vert{}\, \mathbf{z}_{\text{RNCA}}] \in \mathbb{R}^{C_l + 384}$ feeds both a Supervised Contrastive projection head $\mathbf{p} = \text{normalize}(g_{\text{supcon}}(\mathbf{z})) \in \mathbb{S}^{127}$ and the classification head $\hat{\mathbf{y}} = W_{\text{cls}}\text{Dropout}(\text{LayerNorm}(\mathbf{z}))$.

### 5. Composite Logit-Adjusted Focal + Supervised Contrastive Objective
To counter class imbalance (`Normal` $40.79\%$ vs. `Stone` $11.06\%$) while enforcing tight intra-class manifold clustering:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LA-Focal}}\!\left(\hat{\mathbf{y}} + \tau \log \boldsymbol{\pi},\; y\right) + \lambda_{\text{con}}\,\mathcal{L}_{\text{SupCon}}(\mathbf{p}, y)$$

where $\boldsymbol{\pi}$ denotes empirical class priors, $\tau$ is the logit-adjustment temperature, and $\mathcal{L}_{\text{SupCon}}$ pulls same-phenotype embeddings together across patient studies.

### 6. Finite-Sample Split Conformal Prediction
For risk tolerance $\alpha \in (0, 1)$, non-conformity scores $s_i = 1 - \hat{P}(Y = y_i \mid X_i)$ are evaluated on a held-out calibration partition $\mathcal{D}_{\text{cal}}$ of size $n_{\text{cal}}$. Defining the empirical quantile threshold:

$$\hat{q}_{1-\alpha} = \text{Quantile}\!\left(\{s_i\}_{i \in \mathcal{D}_{\text{cal}}},\; \frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}}\right)$$

the prediction set for an unseen patient scan $X_{\text{test}}$ satisfies the finite-sample marginal coverage guarantee $\mathbb{P}\!\left(Y_{\text{test}} \in \mathcal{C}(X_{\text{test}})\right) \ge 1 - \alpha$:

$$\mathcal{C}(X_{\text{test}}) = \left\{ c \in \{1, \dots, 4\} \;:\; 1 - \hat{P}(Y = c \mid X_{\text{test}}) \le \hat{q}_{1-\alpha} \right\}$$

---

## Dataset Integrity & Pseudo-Patient Clustering Audit

A comprehensive cryptographic (`MD5`), geometric, and intensity audit was executed across all **12,446 raw CT images** prior to modeling:

| Audit Metric | Value | Notes |
| :--- | :---: | :--- |
| **Total Raw Images Audited** | `12,446 / 12,446` | 100% valid readable CT slices |
| **True Grayscale Images** | `12,446 / 12,446` | Verified zero-chroma grayscale channels |
| **Exact MD5 Duplicate Files** | `517` | Removed prior to splitting to prevent exact-match leakage |
| **Cross-Class MD5 Collisions** | `0` | Zero label noise across duplicate hashes |
| **Cleaned Leakage-Free Dataset** | `11,929 slices` | Clustered into **427 reconstructed pseudo-patient studies** |
| **Mean Slices per Study** | `27.94 slices/study` | Grouped via `StratifiedGroupKFold(n_splits=5)` |

### Class-Wise Intensity & Geometric Profile ($N = 12,446$)

| Phenotype | Raw Count | Share (%) | Mean Intensity | Std Intensity | Median ($P_{50}$) | $P_{90}$ Intensity | $P_{99}$ Intensity | Aspect Ratio ($H/W$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 5,077 | 40.79% | $44.44 \pm 14.71$ | $65.03 \pm 10.12$ | $5.49 \pm 14.72$ | $142.22 \pm 23.50$ | $248.36 \pm 14.65$ | $1.08 \pm 0.11$ |
| **Cyst** | 3,709 | 29.80% | $26.88 \pm 9.59$ | $49.61 \pm 10.51$ | $0.05 \pm 0.88$ | $92.28 \pm 17.93$ | $227.31 \pm 34.96$ | $1.10 \pm 0.12$ |
| **Tumor** | 2,283 | 18.34% | $36.28 \pm 13.52$ | $60.35 \pm 11.75$ | $0.10 \pm 0.61$ | $125.20 \pm 27.72$ | $244.74 \pm 21.25$ | $1.10 \pm 0.12$ |
| **Stone** | 1,377 | 11.06% | $35.26 \pm 12.98$ | $55.02 \pm 10.79$ | $3.88 \pm 11.26$ | $110.95 \pm 23.24$ | $233.70 \pm 29.56$ | $1.09 \pm 0.11$ |

### 5-Fold `StratifiedGroupKFold` Partition Distribution ($N = 11,929$ Slices / $427$ Studies)

| Fold | Cyst Slices (Studies) | Normal Slices (Studies) | Stone Slices (Studies) | Tumor Slices (Studies) | Total Fold Slices |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 655 (24) | 1,001 (30) | 273 (13) | 457 (18) | **2,386** |
| **Fold 1** | 656 (24) | 1,000 (30) | 271 (10) | 458 (19) | **2,385** |
| **Fold 2** | 656 (24) | 1,001 (30) | 270 (12) | 457 (19) | **2,384** |
| **Fold 3** | 658 (24) | 1,000 (31) | 274 (13) | 456 (19) | **2,388** |
| **Fold 4** | 659 (25) | 1,000 (30) | 272 (13) | 455 (19) | **2,386** |
| **Total** | **3,284 (121)** | **5,002 (151)** | **1,360 (61)** | **2,283 (94)** | **11,929 (427)** |

---

## Experimental Results

### Table 1: Volumetric Slice Data Leakage Audit
Evaluating identical architectures under a standard random slice split versus patient-grouped `StratifiedGroupKFold` exposes severe memorization of adjacent anatomical slices under random splitting:

| Architecture | Random Split Acc (%) | GroupKFold Acc (%) | Acc Inflation ($+\%$) | Random Split MCC | GroupKFold MCC | MCC Inflation ($+$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | 100.00 | 91.05 | $+8.95$ | 1.0000 | 0.8717 | $+0.1283$ |
| **Swin-Tiny** | 99.99 | 97.38 | $+2.61$ | 0.9998 | 0.9625 | $+0.0373$ |
| **ConvNeXt-V2-Tiny** | 100.00 | 96.29 | $+3.71$ | 1.0000 | 0.9469 | $+0.0531$ |
| **PRISM-Net (Proposed)** | 100.00 | 96.63 | $+3.37$ | 1.0000 | 0.9518 | $+0.0482$ |

*(Fold 0 Diagnostic Isolation — PRISM-Net: Random Split achieves $100.00\%$ Acc / $100.00\%$ Macro-F1 / $1.0000$ MCC / $1.0000$ Stone PR-AUC vs. Leakage-Free GroupKFold achieving $95.06\%$ Acc / $93.86\%$ Macro-F1 / $0.9295$ MCC / $0.9816$ Stone PR-AUC.)*

### Table 2: Multi-Model Leakage-Free Benchmark (5-Fold `StratifiedGroupKFold`, $N = 11,929$)

| Architecture | Macro-F1 (%) | MCC | PR-AUC (Cyst) | PR-AUC (Normal) | PR-AUC (Stone) | PR-AUC (Tumor) | Tumor Sens @ 99% Spec (%) | Params (M) | Inference (ms/img) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNetV2-S** | 89.26 | 0.8800 | 0.9588 | 0.9822 | 0.8889 | 0.9426 | 79.33 | 20.18 | 27.25 |
| **ConvNeXt-V2-Tiny** | 95.43 | 0.9469 | 0.9934 | 0.9968 | 0.9810 | 0.9823 | 94.00 | 27.87 | 12.88 |
| **Swin-Tiny** | 97.01 | 0.9625 | 0.9948 | 0.9931 | 0.9914 | 0.9795 | 94.31 | 27.52 | 14.93 |
| **PRISM-Net (Proposed)** | **95.86** | **0.9518** | **0.9916** | **0.9956** | **0.9870** | **0.9858** | **93.91** | **30.69** | **20.22** |

> **Full 5-Fold Out-of-Fold PRISM-Net Summary:**
> * **Overall Accuracy:** `96.63%` | **Macro-F1:** `95.86%` | **MCC:** `0.9518` | **Macro-AUROC:** `0.9968`
> * **Calibration & Proper Scoring:** **ECE:** `0.0322` | **Brier Score:** `0.0558`
> * **Highest Malignant Tumor PR-AUC:** `0.9858` (outperforming all single-stream CNN and ViT baselines on malignant tumor precision-recall discrimination).

### Table 3: Systematic Component Ablation Study (`Folds 0 & 1`, $N = 4,771$ Validation Slices)

| Configuration | Accuracy (%) | Macro-F1 (%) | MCC | PR-AUC (Stone) | Tumor Sens @ 99% Spec (%) | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Dual-Stream Backbone (Duplicated Gray)** | 96.73 | 96.22 | 0.9533 | 0.9900 | 96.60 | 0.0228 |
| **M1: M0 + 3-Channel Pseudo-HU Stem** | 96.63 | 95.92 | 0.9519 | 0.9820 | 96.38 | 0.0247 |
| **M2: M1 + Blind View/Phase FiLM Conditioning** | 97.02 | 96.25 | 0.9575 | 0.9864 | 97.04 | 0.0165 |
| **M3: M2 + Differentiable Radiomic Cross-Attention** | **97.46** | **96.76** | **0.9637** | 0.9879 | 96.71 | **0.0163** |
| **M4: Full PRISM-Net (+ LA-Focal & SupCon Loss)** | 96.61 | 95.75 | 0.9515 | 0.9869 | **97.48** | 0.0275 |

**Key Ablation Takeaways:**
* Adding **Blind View/Phase FiLM Conditioning (`M2`)** reduces Expected Calibration Error by **$33.2\%$** ($0.0247 \rightarrow 0.0165$) while boosting MCC to $0.9575$.
* Integrating **Differentiable Radiomic Cross-Attention (`M3`)** achieves peak discrimination across the ablation ladder (**$97.46\%$ Accuracy**, **$96.76\%$ Macro-F1**, **$0.9637$ MCC**, and **$0.0163$ ECE**).
* Incorporating **Logit-Adjusted Focal + SupCon Loss (`M4`)** maximizes high-specificity clinical triage sensitivity for malignant tumors (**$97.48\%$ Tumor Sensitivity at $99\%$ Specificity**).

### Table 4B: Split Conformal Prediction Clinical Safety Guarantees (5-Fold OOF, $N = 11,929$)

| Risk Level ($\alpha$) | Target Coverage ($1 - \alpha$) | Empirical Coverage (%) | Mean Prediction Set Size $\vert{}\mathcal{C}(X)\vert{}$ | Singleton Fraction (%) | Singleton Accuracy (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 90.0% | **90.08%** | 0.914 | 91.39% | **98.56%** |
| **0.05** | 95.0% | **95.09%** | 0.979 | 97.15% | **97.49%** |
| **0.01** | 99.0% | **98.93%** | 1.112 | 89.15% | **98.90%** |

At the standard clinical operating point ($\alpha = 0.05$), PRISM-Net achieves **$95.09\%$ empirical coverage** while issuing single-class, decisive diagnoses on **$97.15\%$** of all validation scans at **$97.49\%$ accuracy**, automatically deferring only the remaining **$2.85\%$** ambiguous scans for multi-label radiologist review.

---

## Repository & Artifact Structure

```text
CT-KIDNEY-PRISM-Net/
├── notebooks/
│   └── CT_KIDNEY_PRISM_Net.ipynb            # End-to-end training, audit & visualization pipeline
├── prism_artifacts/
│   ├── checkpoints/                         # Cached fold & ablation weights (.pt)
│   │   ├── t1_grp_prism-net_fold_f0..f4.pt
│   │   └── t3_ablation_m0..m3_fold_f0..f1.pt
│   ├── tables/                              # Publication CSV & LaTeX tables
│   │   ├── table1_leakage_audit.{csv,tex}
│   │   ├── table2_sota_benchmark.{csv,tex}
│   │   ├── table3_ablation_study.{csv,tex}
│   │   ├── table4a_causal_saliency.{csv,tex}
│   │   └── table4b_conformal_calibration.{csv,tex}
│   └── figures/                             # 300 DPI publication-grade visualizations
│       ├── fig1_leakage_audit.png
│       ├── fig2_sota_benchmark.png
│       ├── fig3_ablation_trajectory.png
│       ├── fig4_calibration_confusion.png
│       ├── fig5_per_class_causal_saliency.png
│       ├── fig6_roc_pr_curves.png
│       ├── fig7_qualitative_radiomic_saliency.png
│       ├── fig8_raw_vs_predicted_4x5.png
│       ├── fig9_full_pipeline_walkthrough.png
│       ├── fig10_roi_contours_4x5.png
│       ├── fig11_radiomic_feature_distributions.png
│       ├── fig12_conformal_risk_efficiency.png
│       └── fig13_tsne_penultimate_manifold.png
└── README.md

```

---

## Quickstart & Reproducibility

### 1. Environment Setup

```bash
pip install torch torchvision timm scikit-learn pandas numpy opencv-python matplotlib tqdm

```

### 2. Hardware & Training Configuration

* **Compute:** Dual NVIDIA Tesla T4 GPUs ($2 \times 15\text{ GB}$ VRAM) via `nn.DataParallel` with Automatic Mixed Precision (`torch.amp.autocast("cuda")`).
* **Optimization:** `AdamW` ($\text{lr} = 3 \times 10^{-4}$, weight decay $10^{-2}$), linear warmup + Cosine Annealing LR schedule, effective batch size $96$–$128$.
* **Data Hygiene:** Exact `MD5` deduplication ($12,446 \rightarrow 11,929$ slices) followed by perceptual/geometric pseudo-patient clustering ($427$ studies) and 5-fold `StratifiedGroupKFold`.
