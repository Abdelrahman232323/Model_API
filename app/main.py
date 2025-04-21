from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .services.router import ModelRouter
import os

app = FastAPI()
router = ModelRouter()

class UserProfile(BaseModel):
    name: str
    degree: str
    major: str
    gpa: float
    experience: int
    skills: str

@app.on_event("startup")
def load_models_and_data():
    # Load data for each field
    data_path = "data/wuzzuf_02_4_part3.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    # For now, we'll use the same dataset for all fields
    # In production, you'd have separate datasets per field
    for field in router.field_classifier.fields:
        router.load_field_data(field, data_path)

@app.get("/")
def read_root():
    return {"message": "Job Recommendation API with Field Classification"}

@app.post("/recommend")
def recommend_jobs(profile: UserProfile):
    try:
        user_profile_text = f"{profile.degree} in {profile.major}, GPA {profile.gpa}, " \
                           f"{profile.experience} years experience. Skills: {profile.skills}"
        
        recommendations = router.get_recommendations(
            user_profile=user_profile_text,
            top_k=10
        )
        
        return {"recommended_jobs": recommendations}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)