import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load product & user data
def load_data():
    with open(r"C:\Users\navya\OneDrive\Desktop\Customer Experience\Recommendation System\products.json") as f:
        products = json.load(f)
    with open(r"C:\Users\navya\OneDrive\Desktop\Customer Experience\Recommendation System\users.json") as f:
        users = json.load(f)
    return products, users


# Step 1: Rule-based filter (category, brand, price range)
def rule_based_filter(products, user, optional_tags=None):
    prefs = user["preferences"]
    filtered = []
    for p in products:
        if (
            p["category"] in prefs["categories"]
            and p["brand"] in prefs["brands"]
            and prefs["price_range"][0] <= p["price"] <= prefs["price_range"][1]
        ):
            if optional_tags:
                if any(tag in p.get("tags", []) for tag in optional_tags):
                    filtered.append(p)
            else:
                filtered.append(p)
    return filtered


# Step 2: TF-IDF content vector creation
def build_tfidf_matrix(products):
    texts = [p["title"] + " " + p["description"] for p in products]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


# Step 3: Get user preference vector based on viewed products
def get_user_history_vector(user, product_map, vectorizer):
    viewed_texts = [
        product_map[pid]["title"] + " " + product_map[pid]["description"]
        for pid in user["viewed"] if pid in product_map
    ]
    if not viewed_texts:
        return None
    combined_text = " ".join(viewed_texts)
    return vectorizer.transform([combined_text])


# Step 4: Rank products by cosine similarity
def rank_products_by_similarity(filtered_products, tfidf_matrix, user_vector):
    sim_scores = cosine_similarity(user_vector, tfidf_matrix)[0]
    scored = [(filtered_products[i], sim_scores[i]) for i in range(len(filtered_products))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in scored]


# 🔑 MAIN RECOMMENDER FUNCTION
def recommend_for_user(user_id, top_k=5, optional_tags=None):
    products, users = load_data()
    product_map = {p["product_id"]: p for p in products}
    user = next((u for u in users if u["user_id"] == user_id), None)
    if not user:
        return []

    # 1. Rule-based filtering (with optional weather or use-case tags)
    filtered_products = rule_based_filter(products, user, optional_tags)
    if not filtered_products:
        return []

    # 2. Build TF-IDF matrix
    tfidf_matrix, vectorizer = build_tfidf_matrix(filtered_products)

    # 3. Get user vector
    user_vector = get_user_history_vector(user, {p["product_id"]: p for p in filtered_products}, vectorizer)

    # 4. Rank by content similarity
    if user_vector is not None:
        recommended = rank_products_by_similarity(filtered_products, tfidf_matrix, user_vector)
    else:
        recommended = filtered_products  # fallback

    return recommended[:top_k]
