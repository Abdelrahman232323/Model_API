import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class FieldClassifier:
    def __init__(self, model_dir='models'):
        """Initialize the field classifier with predefined job fields"""
        self.fields = ['IT', 'Sales', 'Medical', 'Engineering', 'Finance']
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = MultinomialNB()
        self.model_dir = model_dir
        self.is_trained = False
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def _validate_input(self, text):
        """Validate input text"""
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
    def _check_trained(self):
        """Check if model is trained"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
    def train(self, texts, labels):
        """Train the classifier with job descriptions and their fields"""
        if not texts or not labels or len(texts) != len(labels):
            raise ValueError("Invalid training data: texts and labels must be non-empty and of equal length")
            
        if not all(label in self.fields for label in labels):
            raise ValueError(f"Invalid labels. Must be one of: {self.fields}")
            
        try:
            X = self.vectorizer.fit_transform(texts)
            self.classifier.fit(X, labels)
            self.is_trained = True
            
            # Save the trained model and vectorizer
            joblib.dump(self.vectorizer, os.path.join(self.model_dir, 'vectorizer.joblib'))
            joblib.dump(self.classifier, os.path.join(self.model_dir, 'classifier.joblib'))
        except Exception as e:
            raise RuntimeError(f"Error during training: {str(e)}")
        
    def load_model(self):
        """Load trained model and vectorizer"""
        try:
            vectorizer_path = os.path.join(self.model_dir, 'vectorizer.joblib')
            classifier_path = os.path.join(self.model_dir, 'classifier.joblib')
            
            if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
                self.vectorizer = joblib.load(vectorizer_path)
                self.classifier = joblib.load(classifier_path)
                self.is_trained = True
            else:
                raise FileNotFoundError("Model files not found. Train the model first.")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
        
    def predict(self, text):
        """Predict the job field for given text"""
        self._validate_input(text)
        self._check_trained()
        
        try:
            X = self.vectorizer.transform([text])
            predicted_field = self.classifier.predict(X)[0]
            return predicted_field
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")

    def predict_proba(self, text):
        """Get probability distribution across all fields"""
        self._validate_input(text)
        self._check_trained()
        
        try:
            X = self.vectorizer.transform([text])
            probabilities = self.classifier.predict_proba(X)[0]
            return dict(zip(self.fields, probabilities))
        except Exception as e:
            raise RuntimeError(f"Error during probability prediction: {str(e)}")