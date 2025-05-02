from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model.bert_model import BERTJobRecommender
import os

app = FastAPI()

recommender = None

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
    data_path = "app/data/wuzzuf_02_4_part3.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    recommender = BERTJobRecommender(data_path)

@app.get("/")
def root():
    return {"message": "BERT Job Recommender API is running!"}

@app.post("/recommend")
def recommend_jobs(profile: UserProfile):
    try:
        global recommender
        user_text = f"{profile.degree} in {profile.major}, GPA {profile.gpa}, " \
                    f"{profile.experience} years experience. Skills: {profile.skills}"
        recommendations = recommender.recommend(user_text, top_k=10)
        return {"recommended_jobs": recommendations}
    except Exception as e:
        print(f"[ERROR]: {e}")
        raise HTTPException(status_code=500, detail=str(e))
