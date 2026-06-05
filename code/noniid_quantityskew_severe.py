# ==========================================================
# noniid_quantityskew_severe.py
# Federated Learning under Severe Quantity Skew
# (Data size imbalance, label distribution kept IID)
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
# 2. Reproducibility
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
# 5. Scaling
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
# SEVERE QUANTITY SKEW CLIENT CREATION
# ==========================================================

def create_quantity_skew_clients(X, y):

    """
    Creates 10 clients:
    - 2 large clients (8000 samples each)
    - 8 small clients (1000 samples each)
    Label distribution remains approximately balanced (IID).
    """

    data = np.hstack((X, y.reshape(-1, 1)))
    np.random.shuffle(data)

    clients = {}

    sizes = [8000, 8000] + [1000]*8
    start = 0

    for i, size in enumerate(sizes):
        end = start + size
        chunk = data[start:end]

        X_client = chunk[:, :-1]
        y_client = chunk[:, -1]

        clients[f"client_{i}"] = (X_client, y_client)

        start = end

    return clients


clients = create_quantity_skew_clients(X_train, y_train.values)

print("\nClient sample sizes:")
for name, (X_c, _) in clients.items():
    print(name, ":", X_c.shape[0])


# ==========================================================
# FEDERATED AVERAGING
# ==========================================================

def federated_averaging(clients, X_test, y_test, rounds=40, local_epochs=10):

    global_model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=0.0005,
        random_state=42
    )

    first_client = list(clients.values())[0]
    global_model.partial_fit(first_client[0],
                             first_client[1],
                             classes=np.array([0, 1]))

    accuracy_hist = []
    precision_hist = []
    recall_hist = []
    f1_hist = []
    loss_hist = []

    for r in range(rounds):

        client_weights = []
        client_intercepts = []

        for X_c, y_c in clients.values():

            local_model = SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=0.0005,
                random_state=42
            )

            local_model.coef_ = global_model.coef_.copy()
            local_model.intercept_ = global_model.intercept_.copy()
            local_model.classes_ = global_model.classes_

            for _ in range(local_epochs):
                local_model.partial_fit(X_c, y_c)

            client_weights.append(local_model.coef_)
            client_intercepts.append(local_model.intercept_)

        # Unweighted averaging (intentional)
        global_model.coef_ = np.mean(client_weights, axis=0)
        global_model.intercept_ = np.mean(client_intercepts, axis=0)

        y_pred = global_model.predict(X_test)
        y_proba = global_model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)

        accuracy_hist.append(acc)
        precision_hist.append(precision)
        recall_hist.append(recall)
        f1_hist.append(f1)
        loss_hist.append(loss)

        print(
            f"Round {r+1}: "
            f"Acc={round(acc,4)}, "
            f"Prec={round(precision,4)}, "
            f"Rec={round(recall,4)}, "
            f"F1={round(f1,4)}, "
            f"Loss={round(loss,4)}"
        )

    return accuracy_hist, precision_hist, recall_hist, f1_hist, loss_hist


accuracy_hist, precision_hist, recall_hist, f1_hist, loss_hist = federated_averaging(
    clients,
    X_test,
    y_test
)


# ==========================================================
# FINAL RESULTS
# ==========================================================

print("\nFinal Federated (Severe Quantity Skew) Results:")
print("Accuracy:", round(accuracy_hist[-1], 4))
print("Precision:", round(precision_hist[-1], 4))
print("Recall:", round(recall_hist[-1], 4))
print("F1-score:", round(f1_hist[-1], 4))
print("Loss:", round(loss_hist[-1], 4))


# ==========================================================
# PLOT ACCURACY
# ==========================================================

plt.figure()

plt.plot(range(1, len(accuracy_hist)+1),
         accuracy_hist,
         label="Federated (Quantity Skew)",
         linewidth=2)

plt.axhline(y=central_accuracy,
            linestyle='--',
            linewidth=2,
            label="Centralized Baseline")

plt.xlabel("Communication Rounds")
plt.ylabel("Accuracy")
plt.title("Federated Learning under Severe Quantity Skew")
plt.legend()
plt.grid(True)
plt.show()


# ==========================================================
# PLOT LOSS
# ==========================================================

plt.figure()
plt.plot(range(1, len(loss_hist)+1), loss_hist)
plt.xlabel("Communication Rounds")
plt.ylabel("Log Loss")
plt.title("Loss under Severe Quantity Skew")
plt.grid(True)
plt.show()