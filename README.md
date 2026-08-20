# Urban Heat Island — K-Means Clustering & Interactive Classifier

An unsupervised clustering project on an 11,000-city urban heat island dataset, plus a Flask web app that lets you interactively classify a hypothetical city's thermal profile using the trained K-Means model.

The project has two parts:

1. **`Urban_Heat_Island_KMeans_Clustering.ipynb`** — the full analysis notebook (EDA, preprocessing, optimal-k search, final model, evaluation).
2. **`app.py` + `index.html`** — a Flask app that serves an interactive form: move sliders for 13 city features and get a predicted cluster with a similarity breakdown across all four thermal profiles.

## Demo

The web app presents a dark, editorial-style UI with:
- A left-hand "thesis" panel explaining the model and legend (Urban / Industrial / Green Space / Water).
- A right-hand form with sliders grouped into **Geography**, **Climate**, **Human Activity**, and **Environment & Health**.
- "Load random city" and "Reset to averages" shortcuts.
- A result card showing the predicted cluster, its dominant land cover type, a 4-way similarity spectrum (Water → Green Space → Industrial → Urban), and a per-feature comparison of your input against the cluster's average.

## Project Structure

```
.
├── Urban_Heat_Island_KMeans_Clustering.ipynb   # Analysis notebook
├── urbanheatisland_data.csv                    # Source dataset (not included — see Dataset section)
├── train_model.py                              # Trains & exports model.pkl (see Notes below)
├── model.pkl                                   # Trained scaler + KMeans + cluster profiles (generated)
├── app.py                                      # Flask backend
├── templates/
│   └── index.html                              # Frontend UI
└── README.md
```

## Dataset

- **File:** `urbanheatisland_data.csv`
- **Size:** 11,000 rows × 15 columns
- **Target for evaluation only:** `Land Cover` (Urban, Industrial, Green Space, Water) — not used to fit the model, only to score how well clusters align with reality.
- **13 numeric features used for clustering:**
  - Geography: `Latitude`, `Longitude`, `Elevation (m)`
  - Climate: `Temperature (°C)`, `Wind Speed (km/h)`, `Humidity (%)`, `Annual Rainfall (mm)`
  - Human activity: `Population Density (people/km²)`, `Energy Consumption (kWh)`, `GDP per Capita (USD)`
  - Environment & health: `Air Quality Index (AQI)`, `Urban Greenness Ratio (%)`, `Health Impact (Mortality Rate/100k)`

No missing values in any column.

## Part 1 — Notebook Analysis

### Methodology

1. **EDA** — distributions of temperature/AQI, land cover counts, feature breakdowns by land cover, correlation heatmap.
2. **Preprocessing** — drop `City Name`, standardize the 13 numeric features with `StandardScaler`, label-encode `Land Cover` as ground truth only.
3. **Optimal k search (k = 2–10)** scored with WCSS, Silhouette Score, Calinski-Harabasz Score, and V-Measure.
4. **Final model** — K-Means with **k = 4** (`n_init=10`, `random_state=42`), matching the four land cover categories.
5. **Evaluation** — PCA projection, per-sample silhouette plot, cluster feature profile heatmap, Land Cover × Cluster crosstab.
6. **Exports** — `metrics_by_k.csv`, `final_metrics_summary.csv`, `clustered_output.csv`.

### Key Results

| k | WCSS | Silhouette | Calinski-Harabasz | V-Measure |
|---|------|-----------|--------------------|-----------|
| 2 | 102,017.15 | 0.254 | 4,418.17 | 0.559 |
| 3 | 92,258.79 | 0.180 | 3,024.11 | 0.565 |
| **4** | **84,853.25** | **0.150** | **2,511.72** | **0.545** |
| 5 | 79,939.25 | 0.134 | 2,168.38 | 0.515 |

**Final model (k = 4):** WCSS = 84,853.25 · Silhouette = 0.1495 · Calinski-Harabasz = 2,511.72 · V-Measure = 0.5454

Silhouette and Calinski-Harabasz scores stay low across every tested k, meaning the numeric features don't form sharply separated natural clusters. V-Measure (~0.55) shows moderate — not strong — alignment between the clusters and real `Land Cover` categories. The cluster profiles are still useful for spotting loose tendencies (e.g. hotter/denser vs. cooler/greener), just not for clean classification.

**Possible next steps:** try different feature subsets (e.g. drop lat/long), alternate algorithms (DBSCAN, Agglomerative Clustering, Gaussian Mixture Models), or check whether the dataset is synthetically generated.

## Part 2 — Interactive Flask App

The app loads a pickled bundle (`model.pkl`) containing the fitted `StandardScaler`, the fitted `KMeans` model, the feature column order, per-cluster profiles (dominant land cover, size, land cover breakdown, feature means), and per-feature min/max/mean ranges used to build the sliders.

### How it works

1. `GET /` renders `index.html`, passing `feature_cols` and `feature_ranges` so the frontend can build one slider per feature with sensible bounds.
2. The user adjusts sliders (or clicks "Load random city" / "Reset to averages") and submits the form.
3. `POST /predict` accepts a JSON body of the 13 feature values, scales them with the saved `StandardScaler`, and predicts a cluster with the saved `KMeans` model.
4. Distances to all four centroids are converted into a 0–100 similarity score per cluster.
5. The response includes the predicted cluster, its dominant land cover label and description, confidence, cluster size, land cover breakdown, cluster feature means (for comparison), and similarity scores for all clusters.
6. The frontend renders a result card with a 4-way similarity spectrum and a per-feature "your input vs. cluster average" comparison.

### `model.pkl` bundle format

`app.py` expects a pickle file with this structure:

```python
{
    'scaler': <fitted StandardScaler>,
    'kmeans': <fitted KMeans>,
    'feature_cols': [<13 feature names, in training order>],
    'cluster_profiles': {
        0: {
            'dominant_land_cover': 'Urban',
            'size': 1430,
            'land_cover_breakdown': {...},
            'feature_means': {...}  # unscaled means per feature for this cluster
        },
        ...
    },
    'feature_ranges': {
        'Latitude': {'min': ..., 'max': ..., 'mean': ...},
        ...
    }
}
```

> **Note:** `model.pkl` is generated by a `train_model.py` script (referenced in `app.py`'s header comment but not included here). This script should fit the `StandardScaler` and `KMeans(k=4)` on `urbanheatisland_data.csv` using the same preprocessing as the notebook, then compute and pickle the bundle above. If you don't have this script yet, you can adapt the notebook's Section 4–9 code to build and export it.

## Requirements

```
flask
numpy
pandas
scikit-learn
```

Install with:
```bash
pip install flask numpy pandas scikit-learn
```

(Add `matplotlib`, `seaborn`, and `jupyter` as well if you want to run the notebook.)

## Usage

### Run the notebook
1. Place `urbanheatisland_data.csv` in the same directory as the notebook.
2. Run all cells in order in Jupyter.

### Run the web app
1. Make sure `model.pkl` exists in the project root (see the note above on `train_model.py`).
2. Place `index.html` inside a `templates/` folder (Flask's default template location).
3. Start the server:
   ```bash
   python app.py
   ```
4. Open **http://localhost:8080** in your browser.

## Tech Stack

- **Analysis:** pandas, NumPy, scikit-learn (`KMeans`, `StandardScaler`, `PCA`), matplotlib, seaborn
- **Backend:** Flask
- **Frontend:** vanilla HTML/CSS/JS (no build step), Google Fonts (Space Grotesk, Inter, JetBrains Mono)
