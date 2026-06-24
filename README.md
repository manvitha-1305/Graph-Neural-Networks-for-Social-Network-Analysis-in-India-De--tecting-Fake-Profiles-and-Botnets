# Detection of Fake Profiles and Botnets in Indian Social Networks
## Using Graph Neural Networks (GCN · GAT · GraphSAGE)

**Authors:** Cherukupalli Harshitha, T. Deepthi (Associate Professor)  
**Institution:** Krishna Chaitanya Institute of Technology and Sciences, Markapur

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset (already included)
```bash
python generate_dataset.py
```

### 3. Run the application
```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Usage Guide

### Admin Workflow
1. Go to `http://localhost:5000`
2. Click **Admin Login** → credentials: `admin` / `admin123`
3. Click **Load Dataset** → upload `uploads/Dataset.csv` (or use pre-generated)
4. Click **Run GNN Algorithm** → trains GCN, GAT, and GraphSAGE
5. View accuracy, precision, recall, F-score, confusion matrices, and bar charts
6. Click **View Fake Profiles** to see all user-submitted predictions

### User Workflow
1. Click **New User Signup** → fill in details
2. Login with your credentials
3. Click **Fake Profile Detection**
4. Fill in the 9 profile features and click **Predict**
5. See the GNN result: Genuine ✔ or Fake ⚠ with confidence score

---

## GNN Architecture

| Model | Type | Key Feature |
|-------|------|-------------|
| GCN | Graph Convolutional Network | Spectral graph convolution with symmetric normalization |
| GAT | Graph Attention Network | Attention-weighted neighbor aggregation |
| GraphSAGE | Inductive Learning | Mean-aggregation of sampled neighborhoods |

All three models are implemented **from scratch in NumPy** — no PyTorch or TensorFlow required.

---

## Features Used
- `account_age` — Days since account was created
- `gender` — Male / Female
- `user_age` — Reported age of user
- `link_description` — Whether a link/bio is present (Yes/No)
- `status_count` — Total number of posts/statuses
- `friend_count` — Number of friends/followers
- `internet` — Internet usage metric
- `got_task` — Whether account received coordinated tasks (Yes/No)
- `changed_wifi` — Frequent WiFi/network switching (Yes/No)

---

## Performance (on 1500-record dataset, 90/10 split)
- GCN: ~97% accuracy
- GAT: ~90% accuracy  
- GraphSAGE: ~97% accuracy

Final prediction uses **majority vote** across all three models.
