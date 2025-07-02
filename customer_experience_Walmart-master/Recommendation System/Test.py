from recommender import recommend_for_user

# Simple: Just recommend based on preferences + content
print("Standard Personalized:")
recommendations = recommend_for_user("U001")
for r in recommendations:
    print(f"- {r['title']} ({r['category']})")

# Advanced: Add weather/use-case tag filters
print("\nWeather-aware (rainy):")
weather_recommendations = recommend_for_user("U001", optional_tags=["rain", "cold", "umbrella"])
for r in weather_recommendations:
    print(f"- {r['title']} ({r['tags']})")
