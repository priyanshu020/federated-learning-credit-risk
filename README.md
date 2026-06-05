# Performance Analysis of Federated Learning under Non-IID Financial Tabular Data Distributions

![Experimental Pipeline](figures/experimental_pipeline_flowchart.jpg)

## Overview

This repository contains the implementation, experimental analysis, dissertation report, and presentation for an M.Tech Major Research Project investigating the impact of Non-IID data distributions on Federated Learning performance for financial risk prediction.

The study evaluates the robustness of Federated Learning under different client data heterogeneity scenarios using the UCI Credit Card Default Dataset. Experimental comparisons are conducted between Centralized Learning, FedAvg, and FedProx under IID and Non-IID distributions.

## Research Objectives

The primary objectives of this study were:

- To establish a centralized learning baseline for credit risk prediction.
- To evaluate Federated Learning performance under IID client data distributions.
- To investigate the impact of label skew and quantity skew on Federated Learning.
- To compare the performance of FedAvg and FedProx under heterogeneous data conditions.
- To analyze the suitability of Federated Learning for privacy-preserving financial risk assessment.


## Experimental Setup

### Dataset

- UCI Credit Card Default Dataset
- 30,000 customer records
- Binary classification task (Default / No Default)

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
