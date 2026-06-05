# Performance Analysis of Federated Learning under Non-IID Financial Tabular Data Distributions

![Experimental Pipeline](figures/experimental_pipeline_flowchart.jpg)

## Table of Contents

- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Key Contributions](#key-contributions)
- [Experimental Setup](#experimental-setup)
- [Experimental Results](#experimental-results)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Project Resources](#project-resources)
- [Citation](#citation)
- [Author](#author)

---

## Overview

This repository contains the implementation, experimental analysis, dissertation report, and presentation developed as part of an M.Tech Major Research Project titled **"Performance Analysis of Federated Learning under Non-IID Financial Tabular Data Distributions."**

The study investigates the impact of data heterogeneity on Federated Learning performance using the UCI Credit Card Default Dataset. Various client distribution scenarios, including IID data, moderate label skew, severe label skew, and quantity skew, were evaluated to analyze model robustness under realistic decentralized environments.

The research further compares the performance of **Federated Averaging (FedAvg)** and **Federated Proximal Optimization (FedProx)** under different Non-IID settings and examines their suitability for privacy-preserving financial risk prediction systems.

---

## Research Objectives

The primary objectives of this study were:

- To establish a centralized learning baseline for credit risk prediction.
- To evaluate Federated Learning performance under IID client data distributions.
- To investigate the impact of label skew and quantity skew on Federated Learning.
- To compare the performance of FedAvg and FedProx under heterogeneous data conditions.
- To analyze the suitability of Federated Learning for privacy-preserving financial risk assessment.

---

## Key Contributions

- Implemented a Federated Learning framework for financial risk prediction using tabular credit card data.
- Simulated multiple client data distributions, including IID, 70% label skew, 90% label skew, and severe quantity skew.
- Evaluated the performance of FedAvg and FedProx under varying levels of statistical heterogeneity.
- Performed a comparative analysis against a centralized Logistic Regression baseline.
- Demonstrated the impact of Non-IID data on convergence behavior and predictive performance in Federated Learning systems.

---

## Experimental Setup

### Dataset

- **Dataset:** UCI Credit Card Default Dataset
- **Instances:** 30,000 customer records
- **Task:** Binary Classification (Default / No Default)
- **Domain:** Financial Risk Prediction

### Learning Approaches

- Centralized Logistic Regression
- Federated Averaging (FedAvg)
- Federated Proximal Optimization (FedProx)

### Experimental Scenarios

| Scenario | Algorithm |
|-----------|-----------|
| Centralized Baseline | Logistic Regression |
| IID Distribution | FedAvg |
| Label Skew (70%) | FedAvg, FedProx |
| Label Skew (90%) | FedAvg, FedProx |
| Quantity Skew | FedAvg |

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Log Loss

---

## Experimental Results

### Centralized Baseline

The centralized Logistic Regression model served as the reference benchmark for all Federated Learning experiments.

#### Confusion Matrix

![Confusion Matrix](figures/CM-%20Centralized%20Logistic%20Regression.png)

---

### IID Distribution

Under IID client distributions, Federated Learning achieved performance comparable to centralized learning.

![IID Performance](figures/FL%20(IID)%20vs%20Centralized.png)

---

### Severe Label Skew (90%)

The 90% label skew scenario represented the most challenging Non-IID environment in this study. The results highlight the degradation caused by extreme statistical heterogeneity and demonstrate the relative robustness of FedProx.

#### FedAvg

![FedAvg 90 Accuracy](figures/Accuracy%20under%2090%25%20Label%20Skew%20(FedAvg).png)

#### FedProx

![FedProx 90 Accuracy](figures/Accuracy%20under%2090%25%20Label%20Skew%20(FedProx).png)

---

### Performance Summary

| Scenario | Best Accuracy |
|-----------|-----------|
| Centralized Learning | 80.72% |
| IID Federated Learning | 80.80% |
| Label Skew (70%) - FedAvg | 66.93% |
| Label Skew (70%) - FedProx | 67.02% |
| Label Skew (90%) - FedAvg | 64.38% |
| Label Skew (90%) - FedProx | 64.78% |
| Quantity Skew - FedAvg | 80.80% |

---

## Key Findings

The experimental analysis led to the following observations:

- Federated Learning achieved performance close to centralized learning under IID client distributions.
- Label skew introduced significant performance degradation due to statistical heterogeneity across clients.
- Severe label skew (90%) resulted in the largest reduction in predictive performance.
- Quantity skew had a comparatively smaller impact on model convergence and classification accuracy.
- FedProx demonstrated greater robustness than FedAvg under highly heterogeneous client distributions.
- The results highlight the importance of handling Non-IID data when deploying Federated Learning systems in financial environments.

---

## Repository Structure

```text
federated-learning-credit-risk/
│
├── code/                  # Python implementation
├── dataset/               # UCI Credit Card Dataset
├── figures/               # Experimental plots and flowcharts
├── dissertation/          # Final M.Tech dissertation
├── presentation/          # Final evaluation presentation
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Project Resources

### Dissertation

- Final M.Tech Dissertation Report (PDF and DOCX)

### Presentation

- Final Evaluation Presentation (PPTX and PDF)

### Figures

- Experimental Pipeline Flowchart
- Centralized Confusion Matrix
- IID Performance Analysis
- Label Skew Analysis (70% and 90%)
- Quantity Skew Analysis

---

## Citation

If you use this work in your research, please cite:

```text
Priyanshu Bhardwaj,
"Performance Analysis of Federated Learning under Non-IID Financial Tabular Data Distributions",
M.Tech Dissertation,
University School of Information, Communication and Technology (USICT),
Guru Gobind Singh Indraprastha University (GGSIPU),
2026.
```

---

## Author

**Priyanshu Bhardwaj**

M.Tech (Computer Science & Engineering)

University School of Information, Communication and Technology (USICT)

Guru Gobind Singh Indraprastha University (GGSIPU)

### Research Interests

- Federated Learning
- Explainable AI (XAI)
- Financial Risk Analytics
- Privacy-Preserving Machine Learning
- Healthcare AI

### Contact

GitHub: https://github.com/priyanshu020

Email: priyanshub02031998@gmail.com
