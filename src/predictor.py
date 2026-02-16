"""
Our Own Sentiment Predictor - Uses locally trained model.
No external APIs - completely self-owned!
"""
import os
import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class OurSentimentPredictor:
    """
    Sentiment predictor using our own trained model.
    This is 100% our own - trained by us, running locally!
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize our predictor with our own trained model.
        
        Args:
            model_dir: Directory containing our trained model
        """
        if model_dir is None:
            # Default to models directory - go up from src/ to project root
            model_dir = Path(__file__).parent.parent / 'models'
        else:
            model_dir = Path(model_dir)
        
        self.model_dir = model_dir
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.is_loaded = False
        
        # Load our model
        self._load_model()
    
    def _load_model(self):
        """Load our trained model from disk."""
        
        model_path = self.model_dir / 'sentiment_model.pkl'
        preprocessor_path = self.model_dir / 'preprocessor.pkl'
        metadata_path = self.model_dir / 'metadata.json'
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded our model from: {model_path}")
            
            # Load preprocessor
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info(f"Loaded our preprocessor from: {preprocessor_path}")
            
            # Load metadata
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Model info: {self.metadata}")
            
            self.is_loaded = True
            logger.info("Our sentiment model is ready!")
            
        except FileNotFoundError:
            logger.error("Our model not found! Please run train_model.py first.")
            raise RuntimeError("Model not found. Run train_model.py to create it!")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for input text using our model.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        if not self.is_loaded:
            raise RuntimeError("Our model is not loaded!")
        
        if not text or not text.strip():
            return "neutral", 0.0
        
        # Preprocess the text using our preprocessor
        processed_text = self.preprocessor.preprocess(text)
        
        # Get prediction from our model
        prediction = self.model.predict([processed_text])[0]
        
        # Get confidence scores
        proba = self.model.predict_proba([processed_text])[0]
        confidence = max(proba)
        
        return prediction, confidence
    
    def predict_detailed(self, text: str) -> Dict:
        """
        Get detailed prediction results.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detailed prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Our model is not loaded!")
        
        # Preprocess
        processed_text = self.preprocessor.preprocess(text)
        
        # Get prediction
        prediction = self.model.predict([processed_text])[0]
        
        # Get probabilities
        proba = self.model.predict_proba([processed_text])[0]
        classes = self.model.classes_
        
        # Build probability dict
        probabilities = {}
        for i, cls in enumerate(classes):
            probabilities[cls] = float(proba[i])
        
        result = {
            'text': text,
            'processed_text': processed_text,
            'prediction': prediction,
            'confidence': float(max(proba)),
            'probabilities': probabilities,
            'model_type': 'Our Own TF-IDF + Logistic Regression',
            'model_info': self.metadata
        }
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about our model."""
        return {
            'model_name': 'Our Custom Sentiment Model',
            'model_type': 'TF-IDF + Logistic Regression',
            'is_loaded': self.is_loaded,
            'metadata': self.metadata,
            'description': '100% self-owned, locally trained model'
        }
    
    def add_training_data(self, text: str, label: str):
        """
        Add new training data (for future retraining).
        
        Args:
            text: Training text
            label: Sentiment label ('positive' or 'negative')
        """
        # Store for later retraining
        training_data_file = self.model_dir / 'additional_data.txt'
        
        with open(training_data_file, 'a') as f:
            f.write(f"{label}|{text}\n")
        
        logger.info(f"Added new training data: ({label}) {text[:50]}...")


def create_predictor(model_dir: str = None) -> OurSentimentPredictor:
    """
    Factory function to create our predictor.
    
    Args:
        model_dir: Optional model directory
        
    Returns:
        OurSentimentPredictor instance
    """
    return OurSentimentPredictor(model_dir)

