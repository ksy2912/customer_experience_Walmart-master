import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os

# Load product & user data
def load_data():
    base_path = os.path.dirname(__file__)  # folder where recommender.py lives
    products_path = os.path.join(base_path, "products.json")
    users_path = os.path.join(base_path, "users.json")

    with open(products_path) as f:
        products = json.load(f)
    with open(users_path) as f:
        users = json.load(f)
    return products, users
 

# Step 1: Rule-based filter (by category, brand, price)
def rule_based_filter(products, user):
    filtered = []
    prefs = user["preferences"]
    for p in products:
        if (p["category"] in prefs["categories"] and
            p["brand"] in prefs["brands"] and
            prefs["price_range"][0] <= p["price"] <= prefs["price_range"][1]):
            filtered.append(p)
    return filtered


# Step 2: Content similarity using TF-IDF
def build_tfidf_matrix(products):
    texts = [p["title"] + " " + p["description"] for p in products]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def get_user_history_vector(user, products, vectorizer):
    product_map = {p["product_id"]: p for p in products}
    viewed_texts = [product_map[pid]["title"] + " " + product_map[pid]["description"]
                    for pid in user["viewed"] if pid in product_map]
    if not viewed_texts:
        return None
    user_vector = vectorizer.transform([" ".join(viewed_texts)])
    return user_vector


def rank_products_by_similarity(filtered_products, tfidf_matrix, user_vector, product_indices):
    product_vectors = tfidf_matrix[product_indices]
    sim_scores = cosine_similarity(user_vector, product_vectors)[0]
    scored = [(filtered_products[i], sim_scores[i]) for i in range(len(filtered_products))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored]


# MAIN RECOMMENDER FUNCTION
def recommend_for_user(user_id, top_k=5):
    products, users = load_data()
    user = next((u for u in users if u["user_id"] == user_id), None)
    if not user:
        return []

    # 1. Rule-based filtered products
    filtered = rule_based_filter(products, user)
    if not filtered:
        return []  # No matches after rule filter

    # 2. TF-IDF content-based scoring
    tfidf_matrix, vectorizer = build_tfidf_matrix(filtered)
    user_vector = get_user_history_vector(user, filtered, vectorizer)

    if user_vector is not None:
        recommended = rank_products_by_similarity(filtered, tfidf_matrix, user_vector, list(range(len(filtered))))
    else:
        # No history → fallback to rule-filtered list
        recommended = filtered

    return recommended[:top_k]
