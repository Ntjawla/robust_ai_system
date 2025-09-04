{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Explainability for Sentiment and Intent Models\n",
    "This notebook demonstrates SHAP and LIME explainability for text classification models."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import torch\n",
    "import pandas as pd\n",
    "from transformers import AutoTokenizer, BertForSequenceClassification\n",
    "import shap\n",
    "from lime.lime_text import LimeTextExplainer"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Load sample data\n",
    "df = pd.read_csv('../data/IMDB_dataset.csv')\n",
    "sample_texts = df['review'].tolist()[:5]"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# Load sentiment model\n",
    "sentiment_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)\n",
    "sentiment_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\n",
    "\n",
    "def predict_sentiment(texts):\n",
    "    inputs = sentiment_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)\n",
    "    outputs = sentiment_model(**inputs)\n",
    "    probs = torch.softmax(outputs.logits, dim=1).detach().numpy()\n",
    "    return probs"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# SHAP explainability\n",
    "explainer = shap.Explainer(predict_sentiment, tokenizer=sentiment_tokenizer)\n",
    "shap_values = explainer(sample_texts)\n",
    "shap.plots.text(shap_values[0])"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# LIME explainability\n",
    "class_names = ['negative', 'positive']\n",
    "explainer_lime = LimeTextExplainer(class_names=class_names)\n",
    "\n",
    "for text in sample_texts[:2]:\n",
    "    exp = explainer_lime.explain_instance(text, predict_sentiment, num_features=10)\n",
    "    exp.show_in_notebook(text=True)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
