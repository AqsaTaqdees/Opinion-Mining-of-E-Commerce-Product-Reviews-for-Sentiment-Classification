# Opinion-Mining-of-E-Commerce-Product-Reviews-for-Sentiment-Classification
This project implements a complete Sentiment Classification system for e-commerce product reviews using NLP and IR techniques. Using the Datafiniti Amazon Reviews dataset (71,000+ reviews), the system classifies customer feedback into Happy, OK, or Unhappy categories and retrieves relevant reviews using TF-IDF and BM25 retrieval.

## Project Features
1. End-to-End NLP Pipeline

Text cleaning (punctuation removal, stopword removal, stemming)

Tokenization

TF-IDF vectorization

Sentiment label generation (rating → sentiment)

2. Machine Learning Models

SGD Classifier (Best Performing Model)

Support Vector Machine (SVM)

3. Information Retrieval (IR) Models

BM25 (Okapi) for ranking top-k relevant reviews

TF-IDF cosine similarity retrieval

Side-by-side ranking comparison

4. Evaluation Metrics

Accuracy, Precision, Recall, F1-Score

Retrieval quality comparison (TF-IDF vs BM25)

5. Visualizations

BM25 score plots

TF-IDF vs BM25 top-k comparisons

Classification metric bar charts

## 📊 Key Findings

SGD outperformed SVM across all classification metrics.

BM25 retrieved more relevant top-k reviews than TF-IDF, showing stronger ranking performance.

## 🛠️ Tech Stack
- Python
- Scikit-learn
- NLTK
- Pandas / NumPy
- Rank-BM25
- Matplotlib / Seaborn
- Jupyter Notebook

## 📄 Dataset

Datafiniti’s Consumer Reviews Dataset
Contains: product metadata, ratings, and text reviews.

## 📁 Repository Contents
├── data/                     # Dataset (or link instructions)
├── preprocessing.py          # Text cleaning + sentiment mapping
├── model_training.ipynb      # ML models training & evaluation
├── bm25_retrieval.ipynb      # BM25 indexing & ranking
├── visualizations/           # Plots for analysis
├── README.md                 # Project documentation
└── requirements.txt          # Dependency list
