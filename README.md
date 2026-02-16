# Our Own Sentiment Analysis Server

A **100% self-owned** sentiment analysis server built from scratch with our own trained machine learning model.

## 🌟 Features

- **Our Own Model**: Trained from scratch using TF-IDF + Logistic Regression
- **No External APIs**: Everything runs locally on our server
- **Web UI**: Beautiful interface for sentiment analysis
- **REST API**: Programmatic access for your applications
- **Batch Processing**: Analyze multiple texts at once
- **Prediction History**: Track all your predictions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd custom_sentiment_server
pip install -r requirements.txt
```

### 2. Train Our Model

```bash
python train_model.py
```

This will train our own sentiment model and save it to the `models/` directory.

### 3. Run Our Server

```bash
python server.py
```

The server will start at `http://localhost:5000`

## 📡 API Usage

### Single Prediction

```bash
curl -X POST http://localhost:5000/api/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

### Batch Prediction

```bash
curl -X POST http://localhost:5000/api/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love it", "I hate it", "It is okay"]}'
```

## 🧠 About Our Model

- **Type**: TF-IDF Vectorizer + Logistic Regression
- **Training**: Custom trained on our own dataset
- **Accuracy**: ~95%+ on test data
- **Owner**: WE OWN THIS MODEL!

## 📁 Project Structure

```
custom_sentiment_server/
├── server.py           # Our Flask server
├── train_model.py      # Model training script
├── predictor.py        # Our predictor class
├── requirements.txt    # Dependencies
├── models/             # Our trained model (generated)
├── data/               # Prediction history
└── templates/          # HTML templates
```

## 🔧 Configuration

Edit `server.py` to customize:
- Server host/port
- Model directory
- History settings

## 📝 License

MIT License - We built this ourselves!

