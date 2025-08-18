# app.py
from flask import Flask, render_template, request, jsonify
import joblib, pandas as pd, re, os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# --- Config ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipeline.joblib")
THRESHOLD = 0.5  # HIT if prob >= 0.5

app = Flask(__name__)

# --- Patch missing sklearn internal class (for unpickling portability) ---
import sklearn.compose._column_transformer as _ct
class _RemainderColsList(list): pass
_ct._RemainderColsList = _RemainderColsList

# --- Custom transformer used in training (needed for unpickling) ---
class GenresMultiHot(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vocab_ = None
    def fit(self, X, y=None):
        genres_sets = X["genres_clean"].apply(lambda s: [t for t in s.split(",") if t])
        self.vocab_ = sorted({g for gs in genres_sets for g in gs})
        return self
    def transform(self, X):
        out = np.zeros((len(X), len(self.vocab_)), dtype=int)
        for i, s in enumerate(X["genres_clean"]):
            for g in [t for t in s.split(",") if t]:
                if g in self.vocab_:
                    out[i, self.vocab_.index(g)] = 1
        return out

# --- Load model once ---
pipe = joblib.load(MODEL_PATH)

# Pull the exact genre vocabulary the model knows
# (preprocess step name "prep", transformer name "genres" — matches your Colab)
GENRE_VOCAB = list(getattr(pipe.named_steps["prep"].named_transformers_["genres"], "vocab_", []))
VALID_TYPES = {"movie", "tv show"}

def clean_genres(s: str) -> str:
    toks = [re.sub(r"\s+", " ", t.strip().lower()) for t in str(s).split(",") if t.strip()]
    return ",".join(sorted(set(toks)))

def validate(type_, genres_list, year_str):
    # type
    if type_ not in VALID_TYPES:
        return "Invalid type. Choose Movie or TV Show."
    # year
    if not year_str.isdigit():
        return "Release year must be a number."
    year = int(year_str)
    if year < 1900 or year > 2100:
        return "Release year out of range (1900–2100)."
    # genres (every selected genre must be in vocab)
    bad = [g for g in genres_list if g not in GENRE_VOCAB]
    if bad:
        return f"These genres are not recognized by the model: {', '.join(bad)}"
    return None  # ok

@app.route("/", methods=["GET"])
def home():
    # send vocab to template for the dropdown
    return render_template("index.html", result=None, prob=None, genres_list=GENRE_VOCAB)

@app.route("/predict", methods=["POST"])
def predict():
    title_type   = request.form.get("type", "").strip().lower()
    # because <select multiple>, genres come as a list (possibly empty)
    selected_genres = request.form.getlist("genres")
    selected_genres = [g.strip().lower() for g in selected_genres if g.strip()]
    release_year = request.form.get("release_year", "").strip()

    err = validate(title_type, selected_genres, release_year)
    if err:
        return render_template("index.html", result=f"⚠️ {err}", prob=None, genres_list=GENRE_VOCAB)

    # join into the comma string your pipeline expects
    genres_clean_str = clean_genres(",".join(selected_genres))

    row = pd.DataFrame([{
        "type_clean": title_type,
        "genres_clean": genres_clean_str,
        "release_year": int(release_year),
    }])

    proba = float(pipe.predict_proba(row)[0, 1])
    label = "✅ HIT" if proba >= THRESHOLD else "❌ NOT HIT"

    return render_template("index.html", result=label, prob=f"{proba:.2%}", genres_list=GENRE_VOCAB)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    title_type   = str(data.get("type", "")).strip().lower()
    # allow JSON to send either list or comma string
    genres_value = data.get("genres", [])
    if isinstance(genres_value, str):
        selected_genres = [t.strip().lower() for t in genres_value.split(",") if t.strip()]
    else:
        selected_genres = [str(t).strip().lower() for t in genres_value]
    release_year = str(data.get("release_year", "")).strip()

    err = validate(title_type, selected_genres, release_year)
    if err:
        return jsonify({"error": err}), 400

    genres_clean_str = clean_genres(",".join(selected_genres))
    row = pd.DataFrame([{
        "type_clean": title_type,
        "genres_clean": genres_clean_str,
        "release_year": int(release_year),
    }])

    proba = float(pipe.predict_proba(row)[0, 1])
    return jsonify({"prob_hit": proba, "label": "HIT" if proba >= THRESHOLD else "NOT HIT"})

if __name__ == "__main__":
    app.run(debug=True)
