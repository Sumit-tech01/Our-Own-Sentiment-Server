"""
Text preprocessing utilities for our sentiment model.
"""
import re
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except:
    STOP_WORDS = set()


class TextPreprocessor:
    """Our own text preprocessing pipeline."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = STOP_WORDS
    
    def preprocess(self, text):
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lemmatization
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]

