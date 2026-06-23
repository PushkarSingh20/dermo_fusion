# 🧬 Dermo-Social Pipeline: Multimodal Psoriasis Severity Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch & TensorFlow](https://img.shields.io/badge/Framework-PyTorch%20%26%20TensorFlow-orange.svg)](https://github.com)
[![Optimization: Genetic Algorithm](https://img.shields.io/badge/Optimization-Genetic%20Algorithm-purple.svg)](https://github.com)

An end-to-end multi-stream Artificial Intelligence framework that fuses deep spatial computer vision and transformer-based semantic language embeddings with metaheuristic evolutionary optimization to classify the clinical severity of Psoriasis.

---

## 📋 Table of Contents
* [1. Executive Summary & Clinical Gap](#1-executive-summary--clinical-gap)
* [2. Comparative Academic Positioning](#2-comparative-academic-positioning)
* [3. Mathematical & Theoretical Core](#3-mathematical--theoretical-core)
* [4. System Pipeline Architecture](#4-system-pipeline-architecture)
* [5. Pipeline Implementation Phases](#5-pipeline-implementation-phases)
* [6. Empirical Performance Verification](#6-empirical-performance-verification)
* [7. Repository Structure & Reproducibility](#7-repository-structure--reproducibility)

---

## 🔬 1. Executive Summary & Clinical Gap

Traditional computer-aided diagnostic (CAD) tools in clinical dermatology are inherently mono-modal, relying entirely on visual lesion imagery to evaluate disease stages. However, in real-world clinical practice, the structural manifestation of a skin lesion does not always match the patient's subjective systemic burden or physical suffering. Symptoms such as severe itching, burning, sleep disruption, and psychosocial distress heavily impact patient well-being but leave no explicit visual footprint on the skin.

The **Dermo-Social Pipeline** eliminates this diagnostic blind spot by introducing a **Dual-Stream Multimodal Framework**. It models clinical severity as a composite of:
* **Spatial Clinical Manifestations:** High-resolution geometric and structural textures extracted from lesion photographs.
* **Semantic Patient Narratives:** Subjective, unconstrained textual symptom summaries recorded directly by the patient.

By mapping visual patterns alongside linguistic descriptors, the architecture resolves conflicting signals—such as a localized, visually mild patch paired with severe physical pain—constructing a highly robust, patient-centered clinical decision boundary.

---

## 🎓 2. Comparative Academic Positioning

This project builds upon the evolutionary feature selection paradigms established in state-of-the-art dermatological literature, specifically the hybrid framework proposed by Chauhan & Chauhan (2026) for automated skin lesion identification. While the baseline literature shows that combining dual CNNs with a binary Genetic Algorithm creates a powerful feature reduction model for pure imagery, our framework extends this paradigm into a cross-modal space.

### Framework Evaluation Matrix
| Architectural Dimension | Baseline System (Chauhan & Chauhan, 2026) | **Our Dermo-Social Pipeline** |
| :--- | :--- | :--- |
| **Data Modality** | Mono-Modal (Pure Visual Matrices) | **Multimodal Cross-Domain** (Vision + Semantic Language) |
| **Feature Extractors** | Dual Vision Networks (DenseNet121 + MobileNetV2) | **DenseNet121** (Visual) + **DistilBERT** (Transformer Language Engine) |
| **Data Space** | Homogeneous spatial image descriptors | **Heterogeneous Feature Fusion** (Spatial Geometry + Contextual Token Embeddings) |
| **Optimization Target** | Cross-Validation Accuracy Optimization | **Cost-Sensitive Balanced Accuracy** (Targeting Real-world Minority Class Recall) |
| **Downstream Classifier** | Standard Ensemble Random Forest | **Ensemble Random Forest** with Balanced Class Penalty Weights |

---

## 🧮 3. Mathematical & Theoretical Core

The data processing pipeline is governed by explicit cross-modal math transformations that maintain a rigid feature tracking history.

### Multimodal Feature Fusion Space
Let a clinical input pair consist of an image $I \in \mathbb{R}^{224 \times 224 \times 3}$ and a raw text review string $T$. The spatial feature mapping function $f_V$ and semantic mapping function $f_S$ are defined as:

$$f_V(I) \rightarrow \mathbf{v} \in \mathbb{R}^{1024}$$
$$f_S(T) \rightarrow \mathbf{s} \in \mathbb{R}^{768}$$

The heterogeneous feature fusion matrix $\mathbf{X}_{\text{fused}}$ is constructed by horizontally concatenating the visual descriptor vector and the sentence summary token vector:

$$\mathbf{X}_{\text{fused}} = \left[ \, \mathbf{v} \; \parallel \; \mathbf{s} \, \right] \in \mathbb{R}^{1792}$$

### Metaheuristic Fitness Selection Optimization
To isolate high-contributing biomarkers and eliminate noisy dimensions across the 1792-dimensional space, feature selection is driven by an evolutionary search. Each individual candidate solution is defined as a binary chromosome vector:

$$\mathbf{z} = [z_1, z_2, \dots, z_d] \in \{0, 1\}^d, \quad \text{where } d = 1792$$

To prevent the algorithm from gaming overall accuracy on imbalanced datasets, the fitness function evaluates chromosomes based on **Balanced Accuracy**. This treats errors in rare severe conditions and common mild conditions with equal weight:

$$\text{Fitness}(\mathbf{z}) = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)_{\text{Classifier}(\mathbf{X}[:, \mathbf{z}], \, \mathbf{y})}$$

---

## ⚙️ 4. System Pipeline Architecture

```text
+-----------------------+      +-------------------------+
|  DermNet Image Stream |      | UCI Patient Text Stream |
+-----------+-----------+      +------------+------------+
            |                               |
    [Phase 2: DenseNet121]          [Phase 3: DistilBERT]
            |                               |
    (1024 Visual Features)          (768 Semantic Features)
            |                               |
            +---------------+---------------+
                            |
                 [Phase 4: Feature Fusion]
             (1792-Dimensional Fused Matrix)
                            |
                 [Phase 5: SMOTE Balancing]
             (Synthesized Minority Training Data)
                            |
              [Phase 5: Genetic Optimization]
             (Evolving Best Balanced Feature Mask)
                            |
               [Phase 5: Final Evaluation]
            (Cost-Weighted Random Forest Engine)
