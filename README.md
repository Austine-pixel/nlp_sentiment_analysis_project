# TWEETS SENTIMENT ANALYSIS PROJECT USING NLP & ML

<p align="center">
  <img src="https://img.shields.io/badge/NLP-Tweet%20Classification-blueviolet?style=flat-square"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-LogReg|NB|RF-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Data-1.6M%20Tweets-orange?style=flat-square"/>
</p>

> **Tweet Sentiment Analysis Project** is a scalable, end-to-end NLP pipeline that classifies tweets as **positive** or **negative** using traditional machine learning techniques. Built on the large-scale **Sentiment140** dataset, this project empowers real-time monitoring of customer feedback, social mood, and brand reputation.

---

## Project Overview

This project answers the question:

> **"Can we accurately classify public sentiment from Twitter posts using classical NLP and ML techniques?"**

To solve this, we:
- Cleaned and preprocessed **1.6M+ tweets**
- Explored patterns through **EDA and visualization**
- Built and compared **Logistic Regression**, **Naive Bayes**, and **Random Forest** models
- Evaluated performance through accuracy, precision, recall, F1-score, and error analysis
- Recommend production-ready solutions with potential extensions to deep learning

---

## Dataset Summary

| Column      | Description                            |
|-------------|----------------------------------------|
| `target`    | Sentiment label (0 = Negative, 4 = Positive) |
| `ids`       | Unique tweet ID                        |
| `date`      | Date and time of tweet                 |
| `flag`      | Query parameter (not used)             |
| `user`      | Username of tweeter                    |
| `text`      | Actual tweet content                   |

Source: [Sentiment140 Dataset](http://help.sentiment140.com/for-students)

---

## Tech Stack & Libraries

- **Python 3.10+**
- `nltk`, `re`, `pandas`, `matplotlib`, `seaborn`- for data cleaning & EDA
- `scikit-learn` - vectorization, modeling, evaluation
- `WordCloud`- sentiment word visualizations
- `GridSearchCV` - hyperparameter tuning

---

## EDA & Insights

- Cleaned over **1.6M tweets** using regex, stopword removal, and normalization
- Plotted **tweet length distributions** and **class imbalance**
- Generated **word clouds** and **top keywords** per sentiment class
- Found clear differences in vocabulary and tone between positive and negative tweets

---

## Modeling Approach

We evaluated **three traditional ML models**:

| Model               | Why We Chose It                                   |
|--------------------|----------------------------------------------------|
| **Logistic Regression** | Baseline classifier for linear separation tasks     |
| **Multinomial Naive Bayes** | Simple yet effective for text data, especially with TF-IDF |
| **Random Forest**       | Captures non-linear interactions & provides feature importance |

Each model was evaluated using:
- **Accuracy, Precision, Recall, F1-Score**
- **Confusion Matrices**
- **Feature Importance (Random Forest)**
- **Hyperparameter Tuning (GridSearchCV)**

---

## Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 76.4%    | 75.2%     | 78.9%  | 77.0%    |
| Naive Bayes        | 75.2%    | 74.7%     | 76.4%  | 75.6%    |
| Random Forest      | 75.1%    | 74.9%     | 75.9%  | 75.4%    |

> **Logistic Regression** performed best overall with the highest F1-Score and a strong balance between precision and recall.

---

##  Error Analysis

- Misclassified tweets mostly included:
  - **Sarcasm** or **ambiguous tone**
  - **Short or slang-heavy tweets**
  - **Tweets lacking strong polarity words**
  
We inspected top misclassified examples to guide improvements (e.g.custom tokenizers, sarcasm detection, deep learning).

---

## Validation Strategy

- **Train-test split (80/20)**
- **Stratified sampling** to maintain class balance
- **3-fold cross-validation** via `GridSearchCV` for optimal hyperparameters

---

## Recommendations & Next Steps

- **Deploy the best model (Logistic Regression)** via API (e.g., Flask/FastAPI)
- Add **neutral sentiment** classification
- Integrate **transformer models (e.g.,\ BERT)** for context-aware classification
- Use **streaming data** (e.g., from Twitter API) for real-time dashboards
- Implement **active learning** for continuous model improvement

---

## Acknowledgments

- Dataset by [Alec Go, Richa Bhayani, Lei Huang](http://help.sentiment140.com)
- Inspiration: Stanford NLP Group, Scikit-learn Community
- Special thanks to our mentors and instructurs at Moringa School

---

## Contact & Collaboration
Made with by:

## Pamela Godia
* Email: Pam.Godia@gmail.com 
* LinkedIn: http://linkedin.com/in/pamela-godia-3833b0bb
* Github: https://github.com/PamGodia

## Austine Otieno
* Email: dankotieno1@gmail.com
* LinkedIn: https://www.linkedin.com/in/austine-denis-31792b198/
* Github:https://github.com/Austine-pixel 

## Allan Ofula
* Email: ofulaallan@gmail.com
* LinkedIn: https://www.linkedin.com/in/allan-ofula-b2804911b/
* Github: https://github.com/Allan-Ofula
