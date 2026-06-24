"""
Graph Neural Network (GNN) implementation from scratch using NumPy.
Implements GCN, GAT, and GraphSAGE for fake profile detection.
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Activation helpers
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency(n_nodes, features, k=5):
    """
    Build a k-NN adjacency matrix from node features.
    Nodes that are close in feature space are connected.
    """
    A = np.zeros((n_nodes, n_nodes))
    # Cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-9
    normed = features / norms
    sim = normed @ normed.T          # (n, n)
    # For each node keep top-k neighbours
    for i in range(n_nodes):
        top_k = np.argsort(sim[i])[::-1][1:k+1]
        for j in top_k:
            A[i, j] = 1.0
            A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)        # self-loops
    return A


def normalize_adjacency(A):
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}"""
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D = np.diag(d_inv_sqrt)
    return D @ A @ D


# ─────────────────────────────────────────────────────────────────────────────
# GCN – Graph Convolutional Network
# ─────────────────────────────────────────────────────────────────────────────

class GCN:
    """2-layer Graph Convolutional Network."""

    def __init__(self, in_dim, hidden_dim=64, out_dim=2, lr=0.01, epochs=200, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((in_dim, hidden_dim)) * np.sqrt(2.0 / in_dim)
        self.W2 = rng.standard_normal((hidden_dim, out_dim)) * np.sqrt(2.0 / hidden_dim)
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def forward(self, A_hat, X):
        self.X = X
        self.A_hat = A_hat
        self.Z1 = relu(A_hat @ X @ self.W1)          # (n, hidden)
        self.Z2 = A_hat @ self.Z1 @ self.W2          # (n, out)
        self.out = softmax(self.Z2)
        return self.out

    def _one_hot(self, y, n_classes):
        oh = np.zeros((len(y), n_classes))
        oh[np.arange(len(y)), y.astype(int)] = 1
        return oh

    def fit(self, A_hat, X, y, progress_cb=None):
        n_classes = self.W2.shape[1]
        for epoch in range(self.epochs):
            out = self.forward(A_hat, X)
            Y = self._one_hot(y, n_classes)
            loss = -np.mean(np.sum(Y * np.log(out + 1e-9), axis=1))
            self.losses.append(loss)

            # Back-prop
            dZ2 = (out - Y) / len(y)                         # (n, out)
            dW2 = (self.A_hat @ self.Z1).T @ dZ2             # (hidden, out)
            dZ1 = dZ2 @ self.W2.T                            # (n, hidden)
            dZ1 = self.A_hat @ dZ1
            dZ1[self.Z1 <= 0] = 0                            # ReLU grad
            dW1 = self.X.T @ dZ1                             # (in, hidden)

            self.W2 -= self.lr * dW2
            self.W1 -= self.lr * dW1

            if progress_cb and epoch % 20 == 0:
                progress_cb(epoch, loss)

    def predict_proba(self, A_hat, X):
        return self.forward(A_hat, X)

    def predict(self, A_hat, X):
        return np.argmax(self.predict_proba(A_hat, X), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# GAT – Graph Attention Network (simplified single-head)
# ─────────────────────────────────────────────────────────────────────────────

class GAT:
    """Simplified 2-layer Graph Attention Network."""

    def __init__(self, in_dim, hidden_dim=64, out_dim=2, lr=0.005, epochs=200, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((in_dim, hidden_dim)) * 0.1
        self.a1 = rng.standard_normal((2 * hidden_dim, 1)) * 0.1
        self.W2 = rng.standard_normal((hidden_dim, out_dim)) * 0.1
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def _attention(self, H, A):
        """Compute attention-weighted aggregation."""
        n = H.shape[0]
        # Concat [h_i || h_j] for each edge
        e = np.zeros((n, n))
        for i in range(n):
            hi = H[i]
            concat = np.concatenate([np.tile(hi, (n, 1)), H], axis=1)  # (n, 2h)
            e[i] = (concat @ self.a1).squeeze()
        e = np.where(A > 0, e, -1e9)
        alpha = softmax(e)
        alpha = alpha * A
        return alpha @ H   # aggregated

    def forward(self, A, X):
        H1 = relu(self._attention(X @ self.W1, A))
        self.H1 = H1
        out = softmax(H1 @ self.W2)
        self.out = out
        return out

    def _one_hot(self, y, n_classes):
        oh = np.zeros((len(y), n_classes))
        oh[np.arange(len(y)), y.astype(int)] = 1
        return oh

    def fit(self, A, X, y, progress_cb=None):
        n_classes = self.W2.shape[1]
        for epoch in range(self.epochs):
            out = self.forward(A, X)
            Y = self._one_hot(y, n_classes)
            loss = -np.mean(np.sum(Y * np.log(out + 1e-9), axis=1))
            self.losses.append(loss)

            # Simplified gradient (output layer only for stability)
            dZ2 = (out - Y) / len(y)
            dW2 = self.H1.T @ dZ2
            self.W2 -= self.lr * dW2

            if progress_cb and epoch % 20 == 0:
                progress_cb(epoch, loss)

    def predict_proba(self, A, X):
        return self.forward(A, X)

    def predict(self, A, X):
        return np.argmax(self.predict_proba(A, X), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# GraphSAGE – Inductive Representation Learning
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGE:
    """2-layer GraphSAGE with mean aggregation."""

    def __init__(self, in_dim, hidden_dim=64, out_dim=2, lr=0.01, epochs=200, seed=42):
        rng = np.random.default_rng(seed)
        # Layer-1: concat(self, mean_neigh) -> hidden
        self.W1 = rng.standard_normal((2 * in_dim, hidden_dim)) * np.sqrt(2.0 / (2 * in_dim))
        # Layer-2: concat(self, mean_neigh) -> out
        self.W2 = rng.standard_normal((2 * hidden_dim, out_dim)) * np.sqrt(2.0 / (2 * hidden_dim))
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def _sage_layer(self, A, H, W):
        deg = A.sum(axis=1, keepdims=True)
        deg = np.where(deg == 0, 1, deg)
        neigh_mean = (A @ H) / deg         # mean aggregation
        concat = np.concatenate([H, neigh_mean], axis=1)
        return relu(concat @ W)

    def forward(self, A, X):
        self.X = X
        self.A = A
        H1 = self._sage_layer(A, X, self.W1)
        self.H1 = H1
        H2 = self._sage_layer(A, H1, self.W2)
        out = softmax(H2)
        self.out = out
        self.H2 = H2
        return out

    def _one_hot(self, y, n_classes):
        oh = np.zeros((len(y), n_classes))
        oh[np.arange(len(y)), y.astype(int)] = 1
        return oh

    def fit(self, A, X, y, progress_cb=None):
        n_classes = self.W2.shape[1]
        for epoch in range(self.epochs):
            out = self.forward(A, X)
            Y = self._one_hot(y, n_classes)
            loss = -np.mean(np.sum(Y * np.log(out + 1e-9), axis=1))
            self.losses.append(loss)

            # Gradient for W2
            dH2 = (out - Y) / len(y)
            deg = self.A.sum(axis=1, keepdims=True)
            deg = np.where(deg == 0, 1, deg)
            neigh_h1 = (self.A @ self.H1) / deg
            concat2 = np.concatenate([self.H1, neigh_h1], axis=1)
            dW2 = concat2.T @ dH2
            self.W2 -= self.lr * dW2

            if progress_cb and epoch % 20 == 0:
                progress_cb(epoch, loss)

    def predict_proba(self, A, X):
        return self.forward(A, X)

    def predict(self, A, X):
        return np.argmax(self.predict_proba(A, X), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main GNN pipeline
# ─────────────────────────────────────────────────────────────────────────────

class GNNPipeline:
    """End-to-end pipeline: preprocessing → graph construction → GNN training → evaluation."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.gcn = None
        self.gat = None
        self.sage = None
        self.metrics = {}
        self.is_trained = False
        self.feature_cols = None
        self.A = None
        self.A_hat = None

    # ── feature engineering ──────────────────────────────────────────────────

    FEATURE_COLS = [
        'account_age', 'gender', 'user_age',
        'link_description', 'status_count',
        'friend_count', 'internet', 'got_task',
        'changed_wifi'
    ]

    def _encode(self, df):
        import pandas as pd
        data = df.copy()
        # gender
        if 'gender' in data.columns:
            data['gender'] = data['gender'].map(
                {'Male': 1, 'Female': 0, 'male': 1, 'female': 0,
                 'M': 1, 'F': 0, '1': 1, '0': 0}).fillna(0.5)
        bool_cols = ['link_description', 'got_task', 'changed_wifi']
        for c in bool_cols:
            if c in data.columns:
                data[c] = data[c].map(
                    {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0,
                     'True': 1, 'False': 0, True: 1, False: 0,
                     '1': 1, '0': 0}).fillna(0)
        # internet
        if 'internet' in data.columns:
            data['internet'] = pd.to_numeric(data['internet'], errors='coerce').fillna(0)
        # numeric cols
        num_cols = ['account_age', 'user_age', 'status_count', 'friend_count',
                    'internet', 'status_count']
        for c in num_cols:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors='coerce').fillna(0)
        return data

    def preprocess(self, df):
        """Return X (numpy), y (numpy), and column list."""
        data = self._encode(df)
        available = [c for c in self.FEATURE_COLS if c in data.columns]
        X_raw = data[available].values.astype(float)
        # label
        label_col = None
        for c in ['label', 'fake', 'is_fake', 'class', 'target']:
            if c in data.columns:
                label_col = c
                break
        if label_col:
            y = data[label_col].map(
                {'Fake': 1, 'fake': 1, 'Real': 0, 'real': 0,
                 'Genuine': 0, 'genuine': 0,
                 '1': 1, '0': 0, 1: 1, 0: 0}).fillna(0).values.astype(int)
        else:
            y = np.zeros(len(df), dtype=int)
        self.feature_cols = available
        return X_raw, y

    def fit(self, df, progress_cb=None, max_nodes=300):
        """Train all three GNN architectures."""
        X_raw, y = self.preprocess(df)

        # Subsample for speed if needed
        if len(X_raw) > max_nodes:
            idx = np.random.default_rng(42).choice(len(X_raw), max_nodes, replace=False)
            X_raw, y = X_raw[idx], y[idx]

        X = self.scaler.fit_transform(X_raw)
        in_dim = X.shape[1]

        # Build graph
        A_raw = build_adjacency(len(X), X, k=5)
        self.A = A_raw
        self.A_hat = normalize_adjacency(A_raw)

        # Train / test split (indices only – use same graph)
        idx_all = np.arange(len(X))
        tr, te = train_test_split(idx_all, test_size=0.1, random_state=42, stratify=y)

        def cb(e, l):
            if progress_cb:
                progress_cb(e, l)

        # ── GCN ──
        self.gcn = GCN(in_dim, hidden_dim=32, out_dim=2, lr=0.02, epochs=150)
        self.gcn.fit(self.A_hat, X, y, progress_cb=cb)

        # ── GAT ──
        self.gat = GAT(in_dim, hidden_dim=32, out_dim=2, lr=0.005, epochs=100)
        self.gat.fit(self.A, X, y, progress_cb=cb)

        # ── GraphSAGE ──
        self.sage = GraphSAGE(in_dim, hidden_dim=32, out_dim=2, lr=0.02, epochs=150)
        self.sage.fit(self.A, X, y, progress_cb=cb)

        # ── Evaluate on test nodes ──
        for name, model, A_arg in [
            ('GCN', self.gcn, self.A_hat),
            ('GAT', self.gat, self.A),
            ('GraphSAGE', self.sage, self.A),
        ]:
            prob = model.predict_proba(A_arg, X)
            pred = model.predict(A_arg, X)
            p_te, y_te = pred[te], y[te]
            prob_te = prob[te, 1]
            acc  = accuracy_score(y_te, p_te)
            prec = precision_score(y_te, p_te, zero_division=0)
            rec  = recall_score(y_te, p_te, zero_division=0)
            f1   = f1_score(y_te, p_te, zero_division=0)
            try:
                auc = roc_auc_score(y_te, prob_te)
            except Exception:
                auc = 0.5
            cm = confusion_matrix(y_te, p_te).tolist()
            self.metrics[name] = dict(
                accuracy=round(acc * 100, 2),
                precision=round(prec * 100, 2),
                recall=round(rec * 100, 2),
                f1=round(f1 * 100, 2),
                auc=round(auc * 100, 2),
                confusion_matrix=cm,
                losses=[round(float(l), 4) for l in model.losses[::10]]
            )

        # Store for inference
        self._X_train = X
        self.is_trained = True
        return self.metrics

    def predict_single(self, row_dict):
        """Predict a single user dict."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        import pandas as pd
        df = pd.DataFrame([row_dict])
        data = self._encode(df)
        available = [c for c in self.feature_cols if c in data.columns]
        X_raw = np.zeros((1, len(self.feature_cols)))
        for i, col in enumerate(self.feature_cols):
            if col in data.columns:
                X_raw[0, i] = float(data[col].values[0])
        X = self.scaler.transform(X_raw)

        # Build a small local graph: append to training data
        X_full = np.vstack([self._X_train, X])
        n = len(X_full)
        A = build_adjacency(n, X_full, k=5)
        A_hat = normalize_adjacency(A)

        # Majority vote from all three models
        votes = []
        probs = []
        for model, A_arg in [(self.gcn, A_hat), (self.gat, A), (self.sage, A)]:
            p = model.predict_proba(A_arg, X_full)[-1]
            votes.append(np.argmax(p))
            probs.append(p[1])

        label = 1 if sum(votes) >= 2 else 0
        confidence = round(float(np.mean(probs)) * 100, 1)
        return {
            'label': 'Fake' if label == 1 else 'Genuine',
            'confidence': confidence,
            'votes': {'GCN': int(votes[0]), 'GAT': int(votes[1]), 'GraphSAGE': int(votes[2])},
            'fake_probability': round(float(np.mean(probs)) * 100, 1)
        }

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
