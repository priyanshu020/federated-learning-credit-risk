# ==========================================================
# iid_fedavg.py
# Federated Learning under IID Data Distribution
# Logistic Regression via SGD (Incremental)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss


# -------------------------
# 2. Set Random Seed (FULL REPRODUCIBILITY)
# -------------------------

np.random.seed(42)


# -------------------------
# 3. Load Dataset
# -------------------------

df = pd.read_csv("../dataset/UCI_Credit_Card.csv")

X = df.drop("default.payment.next.month", axis=1)
y = df["default.payment.next.month"]


# -------------------------
# 4. Train-Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# 5. Feature Scaling
# -------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================================
# CENTRALIZED BASELINE (REFERENCE ONLY)
# ==========================================================

central_model = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.0005,
    max_iter=1000,
    random_state=42
)

central_model.fit(X_train, y_train)

central_pred = central_model.predict(X_test)

central_accuracy = accuracy_score(y_test, central_pred)

print("\nCentralized Baseline Accuracy:", round(central_accuracy, 4))


# ==========================================================
# IID CLIENT CREATION
# ==========================================================

def create_iid_clients(X, y, num_clients=10):
    """
    Creates IID client partitions by shuffling
    and splitting the dataset evenly.
    """

    data = np.hstack((X, y.reshape(-1, 1)))

    # Shuffle ensures IID property
    np.random.shuffle(data)

    chunks = np.array_split(data, num_clients)

    clients = {}

    for i, chunk in enumerate(chunks):
        X_c = chunk[:, :-1]
        y_c = chunk[:, -1]
        clients[f"client_{i}"] = (X_c, y_c)

    return clients


clients = create_iid_clients(X_train, y_train.values, num_clients=10)


# ==========================================================
# FEDERATED AVERAGING (FedAvg)
# ==========================================================

def federated_averaging(clients, X_test, y_test, rounds=40, local_epochs=10):

    """
    Performs Federated Averaging under IID setting.
    Tracks:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Log Loss
    """

    # Initialize global model
    global_model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.0005,
        random_state=42
    )

    # Initial partial_fit required for SGDClassifier
    first_client = list(clients.values())[0]
    global_model.partial_fit(first_client[0],
                             first_client[1],
                             classes=np.array([0, 1]))

    # Metric tracking lists
    accuracy_history = []
    precision_history = []
    recall_history = []
    f1_history = []
    loss_history = []

    for r in range(rounds):

        client_weights = []
        client_intercepts = []

        # --------------------------
        # Local Training Phase
        # --------------------------
        for X_c, y_c in clients.values():

            local_model = SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=0.0005,
                random_state=42
            )

            # Initialize local model with global parameters
            local_model.coef_ = global_model.coef_.copy()
            local_model.intercept_ = global_model.intercept_.copy()
            local_model.classes_ = global_model.classes_

            # Perform incremental updates
            for _ in range(local_epochs):
                local_model.partial_fit(X_c, y_c)

            client_weights.append(local_model.coef_)
            client_intercepts.append(local_model.intercept_)

        # --------------------------
        # Aggregation Phase (FedAvg)
        # --------------------------
        global_model.coef_ = np.mean(client_weights, axis=0)
        global_model.intercept_ = np.mean(client_intercepts, axis=0)

        # --------------------------
        # Evaluation Phase
        # --------------------------
        y_pred = global_model.predict(X_test)
        y_proba = global_model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)

        accuracy_history.append(acc)
        precision_history.append(precision)
        recall_history.append(recall)
        f1_history.append(f1)
        loss_history.append(loss)

        print(
            f"Round {r+1}: "
            f"Acc={round(acc,4)}, "
            f"Prec={round(precision,4)}, "
            f"Rec={round(recall,4)}, "
            f"F1={round(f1,4)}, "
            f"Loss={round(loss,4)}"
        )

    return accuracy_history, precision_history, recall_history, f1_history, loss_history


# -------------------------
# Run Federated Training
# -------------------------

accuracy_hist, precision_hist, recall_hist, f1_hist, loss_hist = federated_averaging(
    clients,
    X_test,
    y_test,
    rounds=40,
    local_epochs=10
)


# ==========================================================
# Print Final Round Metrics
# ==========================================================

print("\nFinal Federated (IID) Results:")
print("Accuracy:", round(accuracy_hist[-1], 4))
print("Precision:", round(precision_hist[-1], 4))
print("Recall:", round(recall_hist[-1], 4))
print("F1-score:", round(f1_hist[-1], 4))
print("Loss:", round(loss_hist[-1], 4))


# ==========================================================
# Plot Accuracy vs Centralized Baseline
# ==========================================================

plt.figure()

plt.plot(range(1, len(accuracy_hist)+1),
         accuracy_hist,
         label="Federated (IID)",
         linewidth=2)

plt.axhline(y=central_accuracy,
            linestyle='--',
            linewidth=2,
            label="Centralized Baseline")

plt.xlabel("Communication Rounds")
plt.ylabel("Accuracy")
plt.title("Federated Learning (IID) vs Centralized")
plt.legend()
plt.grid(True)
plt.show()


# ==========================================================
# Plot Loss Curve
# ==========================================================

plt.figure()
plt.plot(range(1, len(loss_hist)+1), loss_hist)
plt.xlabel("Communication Rounds")
plt.ylabel("Log Loss")
plt.title("Federated Learning (IID) - Loss")
plt.grid(True)
plt.show()