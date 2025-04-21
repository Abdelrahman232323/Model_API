# bert_model.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BERTJobRecommender:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.df.dropna(subset=['Job_Title', 'Company_Name', 'Skills', 'Job_Description'], inplace=True)
        self.df['full_text'] = self.df['Job_Title'] + ' at ' + self.df['Company_Name'] + '. ' + \
                               self.df['Job_Description'] + ' Skills: ' + self.df['Skills']
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.embeddings = self.model.encode(self.df['full_text'].tolist(), show_progress_bar=True)

    def recommend(self, user_input: str, top_k: int = 10):
        user_embedding = self.model.encode([user_input])[0]
        similarity_scores = cosine_similarity([user_embedding], self.embeddings)[0]
        self.df['similarity'] = similarity_scores
        top_matches = self.df.sort_values(by='similarity', ascending=False).head(top_k)
        results = top_matches[['Job_Title', 'Company_Name', 'Skills', 'Job_Link', 'similarity']].rename(
            columns={'Job_Link': 'apply_link'}
        )

        # ✅ Convert similarity to float
        results['similarity'] = results['similarity'].astype(float)

        return results.to_dict(orient='records')

        
