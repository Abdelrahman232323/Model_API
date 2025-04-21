from typing import Dict, Type
from ..models.field_classifier.predict import FieldClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

class ModelRouter:
    def __init__(self):
        """Initialize the router with field classifier and model registry"""
        self.field_classifier = FieldClassifier()
        self.models: Dict[str, SentenceTransformer] = {}
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.embeddings: Dict[str, list] = {}
        
    def load_field_data(self, field: str, data_path: str):
        """Load and prepare data for a specific field"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
            
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['Job_Title', 'Company_Name', 'Skills', 'Job_Description'])
        df['combined_text'] = df['Job_Title'] + ' at ' + df['Company_Name'] + '. ' + \
                             df['Job_Description'] + ' Skills: ' + df['Skills']
        
        if field not in self.models:
            self.models[field] = SentenceTransformer('all-MiniLM-L6-v2')
            
        self.datasets[field] = df
        embeddings = self.models[field].encode(
            df['combined_text'].tolist(), 
            show_progress_bar=True
        ).tolist()
        self.embeddings[field] = embeddings
        
    def get_recommendations(self, user_profile: str, field: str | None = None, top_k: int = 10):
        """Get job recommendations for a specific field"""
        if not field:
            field = self.field_classifier.predict(user_profile)
            
        if field not in self.models:
            raise ValueError(f"No model loaded for field: {field}")
            
        user_embedding = self.models[field].encode([user_profile])[0]
        similarities = cosine_similarity(
            [user_embedding], 
            self.embeddings[field]
        )[0]
        
        df = self.datasets[field].copy()
        df['similarity'] = similarities
        top_matches = df.sort_values(by='similarity', ascending=False).head(top_k)
        
        return [
            {
                "job_title": row['Job_Title'],
                "company_name": row['Company_Name'],
                "skills": row['Skills'],
                "apply_link": row.get('Job_Link', ''),
                "similarity_score": round(float(row['similarity']) * 100, 2),
                "field": field
            }
            for _, row in top_matches.iterrows()
        ]