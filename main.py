# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.bert_model import BERTJobRecommender  
import os

app = FastAPI()

# Global model instance
recommender = None

# Pydantic schema for user input
class UserProfile(BaseModel):
    name: str
    degree: str
    major: str
    gpa: float
    experience: int
    skills: str

@app.on_event("startup")
def load_model():
    global recommender
    data_path = "data/wuzzuf_02_4_part3.csv"  
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    recommender = BERTJobRecommender(data_path)

@app.get("/")
def root():
    return {"message": "BERT Job Recommender API is up and running!"}

@app.post("/recommend")
def recommend_jobs(profile: UserProfile):
    try:
        global recommender
        user_text = f"{profile.degree} in {profile.major}, GPA {profile.gpa}, " \
                    f"{profile.experience} years experience. Skills: {profile.skills}"
        recommendations = recommender.recommend(user_text, top_k=10)
        return {"recommended_jobs": recommendations}
    except Exception as e:
        print(f"[ERROR] {str(e)}")  # ✅ Log error in terminal
        raise HTTPException(status_code=500, detail=str(e))


# Run with:
# uvicorn main:app --reload
# Run the app safely on Windows
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
