from recommender import recommend_for_user
import json

results = recommend_for_user("U002", top_k=5)

print("Recommended Products:")
for r in results:
    print(f"- {r['title']} ({r['brand']}) – ₹{r['price']}")
