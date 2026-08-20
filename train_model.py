"""
train_model.py
---------------
Recreates the pipeline from Urban_Heat_Island_KMeans_Clustering.ipynb
and saves a single model.pkl that app.py can load for deployment.

Run this ONCE (locally or in Colab/Jupyter) before running app.py:
    python train_model.py

Requires: urbanheatisland_data.csv to be in the same folder.
"""

import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

RANDOM_STATE = 42
K_FINAL = 4
DATA_PATH = "urbanheatisland_data.csv"

FEATURE_COLS = [
    'Latitude', 'Longitude', 'Elevation (m)', 'Temperature (°C)',
    'Population Density (people/km²)', 'Energy Consumption (kWh)',
    'Air Quality Index (AQI)', 'Urban Greenness Ratio (%)',
    'Health Impact (Mortality Rate/100k)', 'Wind Speed (km/h)',
    'Humidity (%)', 'Annual Rainfall (mm)', 'GDP per Capita (USD)'
]

# ---- 1. Load data (same as notebook) ----
df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLS].values

# ---- 2. Scale features (same as notebook) ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- 3. Ground truth labels (Land Cover) just for cluster interpretation ----
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(df['Land Cover'])

# ---- 4. Fit final KMeans model (k=4, same as notebook) ----
kmeans = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# ---- 5. Build a human-readable profile for each cluster ----
# For every cluster we store: the most common real Land Cover in it (for a
# friendly label), the average of every feature in ORIGINAL units (so the
# webpage can show "your input vs typical value for this cluster"), and how
# many training cities fell into it.
df_out = df.copy()
df_out['Cluster'] = cluster_labels

cluster_profiles = {}
for c in range(K_FINAL):
    subset = df_out[df_out['Cluster'] == c]
    dominant_land_cover = subset['Land Cover'].mode()[0]
    cluster_profiles[c] = {
        "dominant_land_cover": dominant_land_cover,
        "size": int(len(subset)),
        "feature_means": subset[FEATURE_COLS].mean().to_dict(),
        "land_cover_breakdown": subset['Land Cover'].value_counts().to_dict(),
    }

# ---- 6. Package everything the web app needs into ONE pkl file ----
bundle = {
    "scaler": scaler,
    "kmeans": kmeans,
    "feature_cols": FEATURE_COLS,
    "k_final": K_FINAL,
    "cluster_profiles": cluster_profiles,
    "feature_ranges": {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in FEATURE_COLS
    },
}

with open("model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("Saved model.pkl")
print("Cluster summary:")
for c, info in cluster_profiles.items():
    print(f"  Cluster {c}: dominant Land Cover = {info['dominant_land_cover']} "
          f"({info['size']} cities)")
