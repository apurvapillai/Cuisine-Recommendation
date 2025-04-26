import streamlit as st
import joblib
import requests
import os
import json
from collections import defaultdict

# Function to download the model
def download_model(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception("Failed to download the model.")

# URL of the model.pkl file (direct download link)
model_url = "https://drive.google.com/uc?id=11x9H__sytLtprTERNV_OmxczZqn9YYQg"
model_filename = "model.pkl"

# Check if the model file already exists, if not, download it
if not os.path.exists(model_filename):
    download_model(model_url, model_filename)

# Load model + binarizers
model = joblib.load(model_filename)
mlb_ing = joblib.load("mlb_ingredients.pkl")
mlb_cuisine = joblib.load("mlb_cuisine.pkl")

# ── Allowed vegetables ─────────────────────────────────────────────
VEGGIES = [
    "lettuce", "carrot", "onion", "tomato", "spinach", "cucumber", "peas",
    "green beans", "zucchini", "eggplant", "broccoli", "cauliflower",
    "cabbage", "mushroom", "corn", "potato", "asparagus", "celery", "beetroot"
]

# ── Build cuisine→ingredient map for fallback ──────────────────────
with open("train.json") as f:
    raw = json.load(f)

cuisine_ing_map = defaultdict(set)
for r in raw:
    for ing in r["ingredients"]:
        ing_lower = ing.lower()
        for v in VEGGIES:
            if v in ing_lower:
                cuisine_ing_map[r["cuisine"]].add(v)

def fallback(user_ings, k=2):
    best, best_score = None, 0
    for cuisine, ing_set in cuisine_ing_map.items():
        score = len(set(user_ings) & ing_set)
        if score >= k and score > best_score:
            best, best_score = cuisine, score
    return best

# ── Streamlit App Layout ───────────────────────────────────────────
st.title("Cuisine Recommendation App")

# User input for ingredients
chosen = st.multiselect("Select Ingredients", options=VEGGIES)

if st.button("Predict Cuisine"):
    prediction = "No prediction available (try different ingredients)"
    
    if chosen:
        X = mlb_ing.transform([chosen])
        pred_bin = model.predict(X)
        pred_label = mlb_cuisine.inverse_transform(pred_bin)

        if pred_label and pred_label[0]:
            prediction = pred_label[0][0]
        else:  # fallback
            fb = fallback(chosen)
            if fb:
                prediction = fb

    st.success(f"Predicted Cuisine: {prediction}")