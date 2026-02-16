"""
OUR OWN SENTIMENT ANALYSIS SERVER
=================================
A custom Flask server with our own trained model.
No external APIs - everything runs locally on our server!

Features:
- Web UI for sentiment analysis
- REST API endpoints
- Our own ML model (TF-IDF + Logistic Regression)
- 100% self-owned and self-hosted
"""
import os
import json
import csv
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

# Import our own predictor
from predictor import OurSentimentPredictor

# ==================== CONFIGURATION ====================

class Config:
    """Server configuration."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'our-own-secret-key-12345')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Model directory - go up from src/ to project root
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / 'models'
    
    # History settings
    HISTORY_FILE = BASE_DIR / 'data' / 'history.csv'
    HISTORY_LIMIT = 100


# ==================== FLASK APP ====================

app = Flask(__name__)
app.config.from_object(Config)

# Initialize our predictor
predictor = None


def get_predictor():
    """Get or create our predictor instance."""
    global predictor
    if predictor is None:
        print("Loading our own sentiment model...")
        predictor = OurSentimentPredictor(str(Config.MODEL_DIR))
    return predictor


# ==================== HELPER FUNCTIONS ====================

def save_to_history(text: str, sentiment: str, confidence: float):
    """Save prediction to history file."""
    try:
        Config.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        file_exists = Config.HISTORY_FILE.exists()
        
        with open(Config.HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'text', 'sentiment', 'confidence'])
            writer.writerow([
                datetime.now().isoformat(),
                text[:200],
                sentiment,
                f"{confidence:.4f}"
            ])
    except Exception as e:
        print(f"Error saving to history: {e}")


def load_history(limit: int = 100):
    """Load prediction history."""
    predictions = []
    try:
        if Config.HISTORY_FILE.exists():
            with open(Config.HISTORY_FILE, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                all_records = list(reader)
                predictions = all_records[-limit:] if len(all_records) > limit else all_records
    except Exception as e:
        print(f"Error loading history: {e}")
    return predictions


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main page - our sentiment analysis form."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze sentiment from web form."""
    try:
        text = request.form.get('text', '').strip()
        
        if not text:
            flash('Please enter some text to analyze.', 'error')
            return redirect(url_for('index'))
        
        # Use OUR predictor
        p = get_predictor()
        sentiment, confidence = p.predict(text)
        
        # Save to history
        save_to_history(text, sentiment, confidence)
        
        # Format confidence
        confidence_pct = f"{confidence * 100:.1f}%"
        
        return render_template('result.html', 
                             text=text,
                             sentiment=sentiment,
                             confidence=confidence_pct,
                             confidence_raw=confidence)
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    """
    REST API - Our sentiment analysis endpoint.
    Returns JSON response.
    """
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'success': False, 'error': 'Text is required'}), 400
        
        # Use OUR predictor
        p = get_predictor()
        sentiment, confidence = p.predict(text)
        
        # Get detailed prediction
        detailed = p.predict_detailed(text)
        
        # Save to history
        save_to_history(text, sentiment, confidence)
        
        # Return JSON response
        return jsonify({
            'success': True,
            'result': {
                'text': text,
                'sentiment': sentiment,
                'confidence': round(confidence, 4),
                'confidence_percentage': f"{confidence * 100:.1f}%",
                'probabilities': detailed['probabilities'],
                'model': 'Our Own TF-IDF + Logistic Regression'
            }
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sentiment/batch', methods=['POST'])
def api_batch_sentiment():
    """
    Batch API - Analyze multiple texts at once.
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'success': False, 'error': 'texts array is required'}), 400
        
        # Analyze each text with OUR model
        p = get_predictor()
        results = []
        
        for text in texts:
            if text.strip():
                sentiment, confidence = p.predict(text)
                results.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': round(confidence, 4)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/history')
def history():
    """View prediction history."""
    predictions = load_history(Config.HISTORY_LIMIT)
    return render_template('history.html', predictions=predictions)


@app.route('/model-info')
def model_info():
    """Get information about our model."""
    try:
        p = get_predictor()
        info = p.get_model_info()
        return jsonify(info), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        p = get_predictor()
        return jsonify({
            'status': 'healthy',
            'server': 'Our Own Sentiment Server',
            'model_loaded': p.is_loaded,
            'model_type': 'TF-IDF + Logistic Regression',
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/docs')
def api_docs():
    """API documentation."""
    docs = {
        'name': 'Our Own Sentiment Analysis API',
        'description': '100% self-owned server with our own trained model',
        'version': '1.0.0',
        'endpoints': {
            'GET /': 'Web interface',
            'POST /analyze': 'Web form analysis',
            'POST /api/sentiment': 'Single text analysis (JSON)',
            'POST /api/sentiment/batch': 'Batch analysis',
            'GET /history': 'View prediction history',
            'GET /model-info': 'Model information',
            'GET /health': 'Health check',
            'GET /api/docs': 'API documentation'
        },
        'model': {
            'type': 'TF-IDF + Logistic Regression',
            'training': 'Custom trained on our dataset',
            'owner': 'We own this model!'
        }
    }
    return jsonify(docs), 200


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("STARTING OUR OWN SENTIMENT ANALYSIS SERVER")
    print("=" * 60)
    print(f"Server running at: http://{Config.HOST}:{Config.PORT}")
    print("This server uses OUR OWN trained model!")
    print("=" * 60)
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )

