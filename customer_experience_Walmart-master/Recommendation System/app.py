from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from recommender import recommend_for_user

app = FastAPI()

# CORS: Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to frontend URL later
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Walmart Recommender API is live."}

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: str):
    try:
        recommendations = recommend_for_user(user_id, top_k=5)
        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}
