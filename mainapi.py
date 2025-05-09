from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel 
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Models

# Load PCA and trained model
pca = joblib.load('pcaF.pkl')
model = joblib.load('logistic_regression_modelF.pkl')

# Load BERT tokenizer and model
bert_model_path = 'bert_modelM2'
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model = AutoModel.from_pretrained(bert_model_path)

# FastAPI app
app = FastAPI()

# Request body format
class ElaborationInput(BaseModel):
    text: str

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.numpy()

@app.post("/predict_score")
def predict_score(input: ElaborationInput):
    embedding = get_bert_embedding(input.text)
    reduced = pca.transform(embedding)
    prediction = model.predict(reduced)
    return {"score": int(prediction[0])}
