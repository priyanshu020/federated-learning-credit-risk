# ==========================================================
# fedavg_vs_fedprox_labelskew_90.py
# Federated Learning Comparison (90% Label Skew)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss


# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)


# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("../dataset/UCI_Credit_Card.csv")

X = df.drop("default.payment.next.month", axis=1)
y = df["default.payment.next.month"]


# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# Feature Scaling
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================================
# CENTRALIZED BASELINE
# ==========================================================
central_model = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.0005,
    max_iter=1000,
    random_state=42
)

central_model.fit(X_train, y_train)
central_accuracy = accuracy_score(y_test, central_model.predict(X_test))

print("\nCentralized Baseline Accuracy:", round(central_accuracy, 4))


# ==========================================================
# NON-IID CLIENT CREATION (90% LABEL SKEW)
# ==========================================================
def create_noniid_clients(X, y, num_clients=10, skew_ratio=0.9):

    class_0_idx = np.where(y == 0)[0]
    class_1_idx = np.where(y == 1)[0]

    np.random.shuffle(class_0_idx)
    np.random.shuffle(class_1_idx)

    samples_per_client = len(y) // num_clients
    clients = {}

    for i in range(num_clients):

        if i < num_clients // 2:
            major_class = class_0_idx
            minor_class = class_1_idx
        else:
            major_class = class_1_idx
            minor_class = class_0_idx

        major_count = int(samples_per_client * skew_ratio)
        minor_count = samples_per_client - major_count

        selected_major = major_class[:major_count]
        selected_minor = minor_class[:minor_count]

        major_class = major_class[major_count:]
        minor_class = minor_class[minor_count:]

        indices = np.concatenate((selected_major, selected_minor))

        clients[f"client_{i}"] = (X[indices], y[indices])

    return clients


clients = create_noniid_clients(X_train, y_train.values, skew_ratio=0.9)


# ==========================================================
# SHARED INITIAL GLOBAL MODEL
# ==========================================================
base_model = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.0005,
    random_state=42
)

first_client = list(clients.values())[0]
base_model.partial_fit(
    first_client[0],
    first_client[1],
    classes=np.array([0, 1])
)


# ==========================================================
# FEDAVG
# ==========================================================
def federated_fedavg(clients, X_test, y_test, base_model, rounds=40, local_epochs=10):

    global_model = copy.deepcopy(base_model)

    acc_hist, prec_hist, rec_hist, f1_hist, loss_hist = [], [], [], [], []

    for r in range(rounds):

        client_weights, client_intercepts = [], []

        for X_c, y_c in clients.values():

            local_model = copy.deepcopy(global_model)

            for _ in range(local_epochs):
                local_model.partial_fit(X_c, y_c)

            client_weights.append(local_model.coef_)
            client_intercepts.append(local_model.intercept_)

        global_model.coef_ = np.mean(client_weights, axis=0)
        global_model.intercept_ = np.mean(client_intercepts, axis=0)

        y_pred = global_model.predict(X_test)
        y_proba = global_model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)

        acc_hist.append(acc)
        prec_hist.append(prec)
        rec_hist.append(rec)
        f1_hist.append(f1)
        loss_hist.append(loss)

        print(f"[FedAvg] Round {r+1}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, Loss={loss:.4f}")

    return acc_hist, prec_hist, rec_hist, f1_hist, loss_hist


# ==========================================================
# FEDPROX
# ==========================================================
def federated_fedprox(clients, X_test, y_test, base_model, rounds=40, local_epochs=10):

    global_model = copy.deepcopy(base_model)
    mu = 0.1

    acc_hist, prec_hist, rec_hist, f1_hist, loss_hist = [], [], [], [], []

    for r in range(rounds):

        client_weights, client_intercepts = [], []

        for X_c, y_c in clients.values():

            local_model = copy.deepcopy(global_model)

            for _ in range(local_epochs):
                local_model.partial_fit(X_c, y_c)

                # FedProx constraint
                local_model.coef_ -= mu * (local_model.coef_ - global_model.coef_)
                local_model.intercept_ -= mu * (local_model.intercept_ - global_model.intercept_)

            client_weights.append(local_model.coef_)
            client_intercepts.append(local_model.intercept_)

        global_model.coef_ = np.mean(client_weights, axis=0)
        global_model.intercept_ = np.mean(client_intercepts, axis=0)

        y_pred = global_model.predict(X_test)
        y_proba = global_model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)

        acc_hist.append(acc)
        prec_hist.append(prec)
        rec_hist.append(rec)
        f1_hist.append(f1)
        loss_hist.append(loss)

        print(f"[FedProx] Round {r+1}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, Loss={loss:.4f}")

    return acc_hist, prec_hist, rec_hist, f1_hist, loss_hist


# ==========================================================
# RUN BOTH
# ==========================================================
fedavg_acc, fedavg_prec, fedavg_rec, fedavg_f1, fedavg_loss = federated_fedavg(
    clients, X_test, y_test, base_model
)

fedprox_acc, fedprox_prec, fedprox_rec, fedprox_f1, fedprox_loss = federated_fedprox(
    clients, X_test, y_test, base_model
)


# ==========================================================
# FINAL METRICS
# ==========================================================
print("\nFinal FedAvg Results:")
print("Accuracy:", round(fedavg_acc[-1], 4))
print("Precision:", round(fedavg_prec[-1], 4))
print("Recall:", round(fedavg_rec[-1], 4))
print("F1-score:", round(fedavg_f1[-1], 4))
print("Loss:", round(fedavg_loss[-1], 4))

print("\nFinal FedProx Results:")
print("Accuracy:", round(fedprox_acc[-1], 4))
print("Precision:", round(fedprox_prec[-1], 4))
print("Recall:", round(fedprox_rec[-1], 4))
print("F1-score:", round(fedprox_f1[-1], 4))
print("Loss:", round(fedprox_loss[-1], 4))


# ==========================================================
# PLOTS
# ==========================================================

plt.figure()
plt.plot(fedavg_acc, label="FedAvg", linewidth=2)
plt.plot(fedprox_acc, label="FedProx", linewidth=2)
plt.axhline(y=central_accuracy, linestyle='--', label="Centralized")
plt.title("Accuracy Comparison (90% Label Skew)")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


plt.figure()
plt.plot(fedavg_loss, label="FedAvg Loss", linewidth=2)
plt.plot(fedprox_loss, label="FedProx Loss", linewidth=2)
plt.title("Loss Comparison (90% Label Skew)")
plt.xlabel("Rounds")
plt.ylabel("Log Loss")
plt.legend()
plt.grid(True)
plt.show()