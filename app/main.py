from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from app.model_loader import load_artifacts

app = FastAPI(
    title="Product Recommendation Engine",
    description="Hybrid SVD + Diversity recommendation system built on UCI Online Retail II",
    version="1.0.0"
)

# Load once at startup — not on every request
svd, purchase_counts, popularity_scores, user_activity = load_artifacts()

# Precompute once
pop_map = popularity_scores.set_index('StockCode')['purchase_count']
max_pop = pop_map.max()

class RecommendRequest(BaseModel):
    customer_id: int
    n: int = 10
    diversity_weight: float = 0.3

@app.get("/")
def root():
    return {
        "message": "Recommendation Engine is live",
        "docs"   : "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/stats")
def stats():
    return {
        "total_users" : int(purchase_counts['userID'].nunique()),
        "total_items" : int(purchase_counts['itemID'].nunique()),
        "cold_users"  : int((user_activity['interaction_count'] < 5).sum()),
        "warm_users"  : int((user_activity['interaction_count'] >= 5).sum()),
        "sparsity"    : "98.32%",
        "model"       : "SVD (n_factors=100) + Diversity Injection",
        "dataset"     : "UCI Online Retail II (2010-2011)"
    }

@app.post("/recommend")
def recommend(req: RecommendRequest):
    cid = req.customer_id
    n   = req.n

    # Strategy 1: Unknown user
    if cid not in purchase_counts['userID'].values:
        recs = popularity_scores.head(n)[['StockCode', 'description', 'purchase_count']]
        return {
            "customer_id"    : cid,
            "strategy"       : "popularity_fallback_new_user",
            "recommendations": recs.to_dict(orient='records')
        }

    # Strategy 2: Cold user
    interaction_count = user_activity[
        user_activity['userID'] == cid
    ]['interaction_count'].values[0]

    if interaction_count < 5:
        recs = popularity_scores.head(n)[['StockCode', 'description', 'purchase_count']]
        return {
            "customer_id"    : cid,
            "strategy"       : f"popularity_fallback_cold_user_{interaction_count}_interactions",
            "recommendations": recs.to_dict(orient='records')
        }

    # Strategy 3: Warm user — SVD + Diversity
    bought    = set(purchase_counts[purchase_counts['userID'] == cid]['itemID'])
    all_items = purchase_counts['itemID'].unique()
    unseen    = [i for i in all_items if i not in bought]

    preds      = [svd.predict(cid, item) for item in unseen]
    all_scores = np.array([p.est for p in preds])
    norm       = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-9)

    # Score saturation check
    if all_scores.max() - all_scores.min() < 0.01:
        recs = popularity_scores.head(n)[['StockCode', 'description', 'purchase_count']]
        return {
            "customer_id"    : cid,
            "strategy"       : "popularity_fallback_saturation",
            "recommendations": recs.to_dict(orient='records')
        }

    final = []
    for i, pred in enumerate(preds):
        pop_penalty = pop_map.get(pred.iid, 0) / max_pop
        blended     = norm[i] - req.diversity_weight * pop_penalty
        final.append({
            "StockCode"    : pred.iid,
            "svd_score"    : round(pred.est, 3),
            "blended_score": round(float(blended), 3)
        })

    final.sort(key=lambda x: x['blended_score'], reverse=True)

    return {
        "customer_id"    : cid,
        "strategy"       : "svd_diversity_hybrid",
        "recommendations": final[:n]
    }