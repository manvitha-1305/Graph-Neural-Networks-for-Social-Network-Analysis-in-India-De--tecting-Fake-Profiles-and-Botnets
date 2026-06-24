"""
Generate a synthetic Indian social network dataset for fake profile detection.
Saves as Dataset.csv in the uploads directory.
"""
import numpy as np
import pandas as pd
import os

def generate_dataset(n=1500, seed=42):
    rng = np.random.default_rng(seed)

    n_genuine = int(n * 0.60)
    n_fake    = n - n_genuine

    def genuine_users(k):
        return {
            'account_age':      rng.integers(180, 3650, k),          # 6 months – 10 yrs
            'gender':           rng.choice(['Male', 'Female'], k),
            'user_age':         rng.integers(18, 55, k),
            'link_description': rng.choice(['Yes', 'No'], k, p=[0.7, 0.3]),
            'status_count':     rng.integers(50, 5000, k),
            'friend_count':     rng.integers(50, 1500, k),
            'internet':         rng.integers(300, 5000, k),
            'got_task':         rng.choice(['Yes', 'No'], k, p=[0.6, 0.4]),
            'changed_wifi':     rng.choice(['Yes', 'No'], k, p=[0.5, 0.5]),
            'label':            ['Genuine'] * k,
        }

    def fake_users(k):
        return {
            'account_age':      rng.integers(1, 90, k),               # very new
            'gender':           rng.choice(['Male', 'Female'], k),
            'user_age':         rng.choice([0, 99, 100], k),           # unrealistic
            'link_description': rng.choice(['Yes', 'No'], k, p=[0.2, 0.8]),
            'status_count':     rng.integers(0, 20, k),
            'friend_count':     rng.integers(1000, 9999, k),           # abnormally high
            'internet':         rng.integers(9000, 99999, k),
            'got_task':         rng.choice(['Yes', 'No'], k, p=[0.9, 0.1]),
            'changed_wifi':     rng.choice(['Yes', 'No'], k, p=[0.9, 0.1]),
            'label':            ['Fake'] * k,
        }

    g = pd.DataFrame(genuine_users(n_genuine))
    f = pd.DataFrame(fake_users(n_fake))
    df = pd.concat([g, f], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    # Add username column
    usernames = [f"user_{i:04d}" for i in range(len(df))]
    df.insert(0, 'username', usernames)
    return df


if __name__ == '__main__':
    df = generate_dataset()
    out = os.path.join(os.path.dirname(__file__), 'uploads', 'Dataset.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Dataset saved to {out}")
    print(df['label'].value_counts())
