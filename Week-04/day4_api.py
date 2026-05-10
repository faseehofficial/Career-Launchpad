from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sentiment_model')

app = FastAPI(title="Sentiment Analysis API")

if os.path.exists(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
else:
    # Fallback to base model if fine-tuned model doesn't exist yet
    print("Warning: Fine-tuned model not found. Using base DistilBERT.")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: TextRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=-1)
        
    sentiment = "Positive" if prediction.item() == 1 else "Negative"
    
    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=float(confidence.item())
    )

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API. Use /predict to analyze text."}

if __name__ == "__main__":
    import uvicorn
    print("Day 4 Task: Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
