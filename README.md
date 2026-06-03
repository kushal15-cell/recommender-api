# Recommendation Engine API

## Overview

A production-style machine learning recommendation system deployed using FastAPI and Render.

The system combines:

* Collaborative Filtering (SVD)
* Popularity-Based Recommendations
* Cold Start Handling
* Diversity Injection Strategy

The API serves personalized product recommendations for customers based on historical purchase behavior.

---

## Features

### Recommendation Strategies

* New User Fallback
* Cold User Fallback
* Warm User Collaborative Filtering
* Diversity-Aware Recommendations

### API Features

* FastAPI backend
* Interactive Swagger Documentation
* Health Monitoring Endpoint
* Model Statistics Endpoint
* Deployment on Render

---

## Tech Stack

* Python
* FastAPI
* Scikit-Surprise
* Pandas
* NumPy
* Render

---

## API Endpoints

| Endpoint      | Description                |
| ------------- | -------------------------- |
| `/`           | API homepage               |
| `/docs`       | Swagger documentation      |
| `/health`     | Health check               |
| `/stats`      | Dataset & model statistics |
| `/model-info` | Model metadata             |
| `/recommend`  | Generate recommendations   |

---

## Recommendation Logic

### New Users

Users not present in training data receive popularity-based recommendations.

### Cold Users

Users with very few interactions receive popular products.

### Warm Users

Users with sufficient interactions receive personalized SVD collaborative filtering recommendations with diversity adjustment.

---

## Model Architecture

### Collaborative Filtering

* SVD matrix factorization
* Latent feature learning
* User-item interaction modeling

### Diversity Injection

A popularity penalty is applied to avoid recommending only highly popular products.

Final score:

blended_score = normalized_svd_score - diversity_weight * popularity_penalty

---

## Dataset

UCI Online Retail II Dataset (2010–2011)

Contains:

* Customer transactions
* Product purchases
* Invoice data
* Product descriptions

---

## Deployment

Deployed on Render using FastAPI and Uvicorn.

---

## Future Improvements

* Real-time recommendations
* Approximate nearest neighbors
* Hybrid content-based filtering
* User embeddings
* A/B testing
* Redis caching

---

## Author

Kushal K Melagiri
