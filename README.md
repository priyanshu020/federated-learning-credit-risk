# Performance Analysis of Federated Learning under Non-IID Financial Tabular Data Distributions

## Overview

This repository contains the implementation and experimental results of an M.Tech dissertation focused on evaluating the impact of Non-IID data distributions on Federated Learning performance for financial risk prediction.

The study investigates how different client data distributions affect model convergence and predictive performance and compares FedAvg and FedProx under heterogeneous environments.

## Dataset

UCI Credit Card Default Dataset

## Experimental Scenarios

- Centralized Baseline
- Federated Learning (IID)
- Label Skew (70%)
- Label Skew (90%)
- Quantity Skew
- FedAvg vs FedProx Comparison

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Log Loss

## Key Findings

- Federated Learning achieved performance comparable to centralized learning under IID conditions.
- Label skew significantly degraded performance.
- Quantity skew had minimal impact on model performance.
- FedProx demonstrated improved robustness under severe heterogeneity.

## Author

Priyanshu Bhardwaj

M.Tech (Computer Science & Engineering)

USICT, GGSIPU
