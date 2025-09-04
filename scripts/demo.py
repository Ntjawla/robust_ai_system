import torch
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline
from utils import log_prediction
from rag_index import retrieve_context

# ----------------------------
# Load Models
# ----------------------------
# Sentiment Agent
sentiment_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
sentiment_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Intent Agent (dummy example: 3 classes)
intent_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
intent_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Summarization Agent
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ----------------------------
# Multi-Agent Prediction
# ----------------------------
def multi_agent_predict(review: str):
    # Retrieve context from KB
    context = retrieve_context(review)
    
    # Sentiment
    inputs = sentiment_tokenizer([review], return_tensors="pt", truncation=True, padding=True)
    outputs = sentiment_model(**inputs)
    sentiment = 'positive' if torch.argmax(outputs.logits) == 1 else 'negative'
    
    # Intent
    inputs_intent = intent_tokenizer([review], return_tensors="pt", truncation=True, padding=True)
    intent_outputs = intent_model(**inputs_intent)
    intent = torch.argmax(intent_outputs.logits).item()
    
    # Summarization
    summary = summarizer(review, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
    
    # Log prediction
    log_prediction(review, sentiment, intent, summary)
    
    return {"review": review, "sentiment": sentiment, "intent": intent, "summary": summary, "context": context}

# ----------------------------
# Run Demo on Sample Reviews
# ----------------------------
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

reviews_file = os.path.join(PROJECT_ROOT, "data", "sample_reviews.txt")
with open(reviews_file, "r", encoding="utf-8") as f:
    sample_reviews = f.read().splitlines()


for review in sample_reviews:
    result = multi_agent_predict(review)
    print("\n--- Prediction ---")
    print(f"Review: {result['review']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Intent: {result['intent']}")
    print(f"Summary: {result['summary']}")
    print(f"Knowledge Context: {result['context']}")
