from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# ---------------------------------------------------------------------------
# Load the trained pipeline (StandardScaler + KMeans + cluster profiles)
# Generate this file first by running train_model.py
# ---------------------------------------------------------------------------
with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)

scaler = bundle['scaler']
kmeans = bundle['kmeans']
FEATURE_COLS = bundle['feature_cols']
CLUSTER_PROFILES = bundle['cluster_profiles']
FEATURE_RANGES = bundle['feature_ranges']

# Short descriptions shown on the result card, keyed by the dominant
# Land Cover type discovered for each cluster during training.
LAND_COVER_DESCRIPTIONS = {
    "Urban": "Dense built-up areas with high population density and energy "
             "consumption, elevated temperature and reduced greenness — the "
             "classic urban heat island signature.",
    "Industrial": "High energy consumption and pollution levels (AQI), "
                  "moderate-to-high temperature, and lower urban greenness.",
    "Green Space": "Higher urban greenness ratio and lower AQI, with cooler "
                   "temperatures and a healthier environmental profile.",
    "Water": "Locations near/around water bodies — typically cooler "
              "temperatures, higher humidity and lower heat retention.",
}

app = Flask(__name__)


def build_feature_order():
    """Return the form field names in the exact order the model expects."""
    return FEATURE_COLS


@app.route('/')
def index():
    return render_template(
        'index.html',
        feature_cols=FEATURE_COLS,
        feature_ranges=FEATURE_RANGES,
    )


@app.route('/predict', methods=['POST'])
def predict_cluster():
    try:
        data = request.get_json(silent=True) or request.form

        # Read the 13 features in the exact order used for training
        values = []
        for col in FEATURE_COLS:
            raw = data.get(col)
            if raw is None or raw == '':
                return jsonify({
                    'error': f"Missing value for '{col}'."
                }), 400
            values.append(float(raw))

        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        cluster = int(kmeans.predict(X_scaled)[0])

        # Distance of this point to every cluster centroid, turned into a
        # 0-100 "similarity" score so the page can show a confidence-style bar
        distances = kmeans.transform(X_scaled)[0]
        inv = 1.0 / (1.0 + distances)
        similarity = (inv / inv.sum()) * 100

        profile = CLUSTER_PROFILES[cluster]
        dominant = profile['dominant_land_cover']

        response = {
            'cluster': cluster,
            'dominant_land_cover': dominant,
            'description': LAND_COVER_DESCRIPTIONS.get(dominant, ''),
            'confidence': round(float(similarity[cluster]), 1),
            'cluster_size': profile['size'],
            'land_cover_breakdown': profile['land_cover_breakdown'],
            'input_values': dict(zip(FEATURE_COLS, values)),
            'cluster_means': profile['feature_means'],
            'all_clusters': [
                {
                    'cluster': c,
                    'dominant_land_cover': CLUSTER_PROFILES[c]['dominant_land_cover'],
                    'similarity': round(float(similarity[c]), 1),
                }
                for c in range(len(CLUSTER_PROFILES))
            ],
        }
        return jsonify(response)

    except ValueError:
        return jsonify({'error': 'Please enter valid numeric values for every field.'}), 400
    except Exception as e:
        return jsonify({'error': f'Something went wrong: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
