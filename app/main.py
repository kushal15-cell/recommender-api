from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from app.model_loader import load_artifacts


app = FastAPI(
    title="Recommendation Engine API",
    description="""
    A deployed machine learning recommendation system using
    SVD collaborative filtering with popularity-based fallback strategies.
    """,
    version="1.0.0"
)


# Load artifacts once when the API starts
svd, purchase_counts, popularity_scores, user_activity = load_artifacts()

# Precompute useful lookup values
pop_map = popularity_scores.set_index("StockCode")["purchase_count"]
max_pop = pop_map.max()
all_items = purchase_counts["itemID"].unique()
known_users = set(purchase_counts["userID"].unique())


class RecommendRequest(BaseModel):
    customer_id: int = Field(
        ...,
        example=12347,
        description="Customer ID for whom recommendations are generated"
    )
    n: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of recommendations to return"
    )
    diversity_weight: float = Field(
        0.3,
        ge=0,
        le=1,
        description="Controls how much popularity penalty is applied"
    )


@app.get("/", tags=["General"])
def home():
    return {
        "project": "Recommendation Engine API",
        "status": "live",
        "description": "ML-powered product recommendation API",
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats",
        "model_info": "/model-info",
        "recommendation_endpoint": "/recommend",
        "version": "1.0.0"
    }


@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "message": "Recommendation API is running successfully"
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_type": "Hybrid Recommendation System",
        "algorithm": "SVD Collaborative Filtering + Popularity Fallback",
        "dataset": "UCI Online Retail II",
        "cold_start_strategy": "Popularity-based recommendations",
        "deployment": "Render",
        "framework": "FastAPI"
    }


@app.get("/stats", tags=["Model"])
def stats():
    return {
        "total_users": int(purchase_counts["userID"].nunique()),
        "total_items": int(purchase_counts["itemID"].nunique()),
        "cold_users": int((user_activity["interaction_count"] < 5).sum()),
        "warm_users": int((user_activity["interaction_count"] >= 5).sum()),
        "sparsity": "98.32%",
        "model": "SVD with diversity adjustment",
        "dataset": "UCI Online Retail II"
    }


@app.post(
    "/recommend",
    tags=["Recommendations"],
    summary="Generate product recommendations",
    description="""
    Generates personalized product recommendations for a customer.

    Recommendation strategy:
    - New user: popularity-based fallback
    - Cold user: popularity-based fallback
    - Warm user: SVD collaborative filtering with diversity penalty
    """
)
def recommend(req: RecommendRequest):
    customer_id = req.customer_id
    n = req.n

    # Case 1: Unknown/new user
    if customer_id not in known_users:
        recs = popularity_scores.head(n)[
            ["StockCode", "description", "purchase_count"]
        ]

        return {
            "customer_id": customer_id,
            "strategy": "popularity_fallback_new_user",
            "message": "Customer not found, returning popular products",
            "recommendations": recs.to_dict(orient="records")
        }

    # Get customer interaction count
    customer_activity = user_activity[
        user_activity["userID"] == customer_id
    ]

    if customer_activity.empty:
        raise HTTPException(
            status_code=404,
            detail="Customer activity data not found"
        )

    interaction_count = int(
        customer_activity["interaction_count"].values[0]
    )

    # Case 2: Cold user
    if interaction_count < 5:
        recs = popularity_scores.head(n)[
            ["StockCode", "description", "purchase_count"]
        ]

        return {
            "customer_id": customer_id,
            "strategy": "popularity_fallback_cold_user",
            "interaction_count": interaction_count,
            "message": "Customer has limited history, returning popular products",
            "recommendations": recs.to_dict(orient="records")
        }

    # Case 3: Warm user — SVD + diversity
    bought_items = set(
        purchase_counts[
            purchase_counts["userID"] == customer_id
        ]["itemID"]
    )

    unseen_items = [
        item for item in all_items
        if item not in bought_items
    ]

    if len(unseen_items) == 0:
        recs = popularity_scores.head(n)[
            ["StockCode", "description", "purchase_count"]
        ]

        return {
            "customer_id": customer_id,
            "strategy": "popularity_fallback_no_unseen_items",
            "message": "No unseen items available for this customer",
            "recommendations": recs.to_dict(orient="records")
        }

    predictions = [
        svd.predict(customer_id, item)
        for item in unseen_items
    ]

    raw_scores = np.array([pred.est for pred in predictions])

    score_range = raw_scores.max() - raw_scores.min()

    # If model scores are almost identical, fallback to popularity
    if score_range < 0.01:
        recs = popularity_scores.head(n)[
            ["StockCode", "description", "purchase_count"]
        ]

        return {
            "customer_id": customer_id,
            "strategy": "popularity_fallback_score_saturation",
            "message": "Model scores were too similar, returning popular products",
            "recommendations": recs.to_dict(orient="records")
        }

    normalized_scores = (
        (raw_scores - raw_scores.min()) /
        (score_range + 1e-9)
    )

    final_recommendations = []

    for i, pred in enumerate(predictions):
        popularity_penalty = pop_map.get(pred.iid, 0) / max_pop

        blended_score = (
            normalized_scores[i]
            - req.diversity_weight * popularity_penalty
        )

        final_recommendations.append({
            "StockCode": pred.iid,
            "svd_score": round(float(pred.est), 3),
            "blended_score": round(float(blended_score), 3)
        })

    final_recommendations = sorted(
        final_recommendations,
        key=lambda x: x["blended_score"],
        reverse=True
    )

    return {
        "customer_id": customer_id,
        "strategy": "svd_diversity_hybrid",
        "interaction_count": interaction_count,
        "recommendations": final_recommendations[:n]
    }