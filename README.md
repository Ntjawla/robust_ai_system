# Robust AI System: AI/ML & Data Engineering Portfolio Project

This project demonstrates a **production-ready AI/ML pipeline** combining **sentiment analysis, intent classification, summarization, and knowledge retrieval (RAG)**, with explainability using **SHAP and LIME**.

It showcases skills in:

- **AI/ML:** Hugging Face Transformers (DistilBERT, BART), LLM pipelines, sentiment & summarization models.
- **Data Engineering:** ETL, FAISS-based knowledge retrieval, embeddings with SentenceTransformers.
- **Explainability:** SHAP and LIME for model interpretability.
- **Cloud/Deployment Awareness:** Modular Python scripts, virtual environment management, reproducible pipelines.

---

## Key Features

- Multi-agent pipeline: Sentiment, intent, summarization, and knowledge retrieval.  
- Production-grade RAG with FAISS + embeddings.  
- Explainable AI with SHAP & LIME.  
- Clean, readable outputs suitable for enterprise demos.  

---

## Quick Start


```bash
git clone https://github.com/Ntjawla/robust_ai_system
cd robust_ai_system


## Create a virtual environment

python -m venv ai_env
ai_env\Scripts\activate   # Windows
pip install --upgrade pip
pip install -r requirements.txt


## Run demo :  

python scripts\demo.py


## View explainability notebook : 

jupyter notebook notebooks/explainability.ipynb

