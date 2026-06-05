# ==========================================================
# baseline.py
# Centralized Logistic Regression (SGD) Benchmark
# ==========================================================

# -------------------------
# 1. Import Libraries
# -------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# -------------------------
# 2. Set Random Seed (Reproducibility)
# -------------------------

# Ensures that dataset shuffling and model training
# produce identical results across multiple runs
np.random.seed(42)


# -------------------------
# 3. Load Dataset
# -------------------------

# UCI Credit Card Default Dataset
df = pd.read_csv("../dataset/UCI_Credit_Card.csv")

# Separate features and target variable
X = df.drop("default.payment.next.month", axis=1)
y = df["default.payment.next.month"]


# -------------------------
# 4. Train-Test Split
# -------------------------

# 80-20 split with stratification to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# 5. Feature Scaling
# -------------------------

# Standardize features to improve convergence
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training samples:", X_train.shape)
print("Test samples:", X_test.shape)


# ==========================================================
# CENTRALIZED MODEL TRAINING
# ==========================================================

# Logistic Regression implemented using SGD
# 'log_loss' enables probabilistic logistic regression behaviour
model = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.0005,       # learning rate used in experiments
    max_iter=1000,
    random_state=42
)

# Train centralized model
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)


# -------------------------
# 6. Evaluation Metrics
# -------------------------

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nCentralized Logistic Regression Results")
print("Accuracy:", round(acc, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))


# ==========================================================
# 7. Confusion Matrix
# ==========================================================

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# ==========================================================
# 8. Plot Confusion Matrix (For Dissertation Figure)
# ==========================================================

# Visualization of classification performance
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix- Centralized Logistic Regression")

# Save high-resolution figure for dissertation
plt.savefig("confusion_matrix_baseline.png", dpi=300)
plt.show()