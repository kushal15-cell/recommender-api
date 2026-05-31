import pickle
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "models"

def load_artifacts():
    with open(BASE / "svd_v2_model.pkl", "rb") as f:
        svd = pickle.load(f)
    with open(BASE / "purchase_counts_v2.pkl", "rb") as f:
        purchase_counts = pickle.load(f)
    with open(BASE / "popularity_scores.pkl", "rb") as f:
        popularity_scores = pickle.load(f)
    with open(BASE / "user_activity.pkl", "rb") as f:
        user_activity = pickle.load(f)
    return svd, purchase_counts, popularity_scores, user_activity