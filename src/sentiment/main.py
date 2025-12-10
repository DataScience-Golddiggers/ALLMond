from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import os

app = FastAPI()

MODEL_PATH = "/app/models/sentiment"
BASE_MODEL_NAME = "distilbert-base-uncased"

class SentimentRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        
        # Define labels mapping
        id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
        
        if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "adapter_config.json")):
            print(f"Loading LoRA model from {MODEL_PATH}")
            # Load base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL_NAME, 
                num_labels=3,
                id2label=id2label,
                label2id=label2id
            )
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        else:
            print("LoRA model not found. Loading base model for testing (untrained).")
            model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL_NAME,
                num_labels=3,
                id2label=id2label,
                label2id=label2id
            )
        
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
            
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/analyze", response_model=List[SentimentResponse])
async def analyze_sentiment(request: SentimentRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for text in request.texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][prediction].item()
                
                # Access id2label from the base model config if wrapped in PeftModel
                config = model.base_model.model.config if isinstance(model, PeftModel) else model.config
                sentiment_label = config.id2label[prediction]
                
                results.append({
                    "text": text,
                    "sentiment": sentiment_label,
                    "confidence": confidence
                })
        except Exception as e:
            print(f"Error analyzing text: {e}")
            results.append({
                "text": text,
                "sentiment": "Error",
                "confidence": 0.0
            })
            
    return results

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
