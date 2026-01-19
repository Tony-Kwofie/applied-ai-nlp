# Explainable NLP for Injury Severity Prediction  
**Applied Artificial Intelligence | MSc Artificial Intelligence & Data Science**  
**University of Hull (London Campus)**

---

## 1. Project Overview
This project investigates the use of **Natural Language Processing (NLP)** and **Explainable Artificial Intelligence (XAI)** to predict **injury severity** from free-text occupational accident narratives. The work is situated within **safety-critical industrial contexts**, where transparency, interpretability, and trust are essential for real-world AI adoption.

Rather than focusing solely on predictive accuracy, this study emphasises **human-centred and interpretable AI**, enabling safety professionals to understand *why* a model predicts severe outcomes and which linguistic patterns contribute most to risk.

---

## 2. Research Motivation
Occupational Health & Safety (OHS) systems generate large volumes of unstructured incident reports, yet much of this data remains under-analysed. Black-box AI models are often unsuitable in such domains due to regulatory, ethical, and operational constraints.

This project addresses the research gap by:
- Applying NLP to real-world injury narratives  
- Balancing predictive performance with explainability  
- Demonstrating how XAI can support decision-making in safety-critical environments  

---

## 3. Research Objectives
- Predict injury severity using NLP-based machine learning models  
- Compare interpretable classical models with deep learning approaches  
- Address real-world challenges such as noisy text and class imbalance  
- Extract meaningful linguistic patterns linked to severe injuries  
- Explore the role of explainability in trustworthy safety analytics  

---

## 4. Methodology

### 4.1 Data and Preprocessing
- Real-world occupational injury narratives (OSHA-style reports)
- Text cleaning, normalisation, and tokenisation
- Narrative length analysis and exploratory text statistics
- Handling class imbalance using SMOTE

### 4.2 Machine Learning Models
- TF-IDF feature extraction
- Logistic Regression with resampling
- Evaluation metrics:
  - Precision–Recall
  - ROC–AUC
  - F1-score
  - Confusion Matrix

### 4.3 Deep Learning Models
- Bidirectional Long Short-Term Memory (BiLSTM)
- Sequential modelling of injury narratives
- Analysis of training dynamics and generalisation behaviour

---

## 5. Explainable AI (XAI) Focus
Explainability is a central contribution of this project. The study incorporates multiple interpretability-oriented analyses, including:

- Identification of high-impact words and phrases associated with severe injuries  
- Class-specific keyword analysis and word clouds  
- Feature importance interpretation for linear models  
- Comparative analysis between interpretable ML models and deep neural networks  

This approach supports **trustworthy AI**, allowing domain experts to validate, question, and learn from model outputs.

---

## 6. Results and Analysis
The repository includes extensive quantitative and visual evaluation:

- Confusion matrices and classification reports  
- Precision–Recall and ROC curves  
- F1-score comparisons across models  
- Narrative length distribution analysis  
- Word clouds for hospitalized vs non-hospitalized cases  
- BiLSTM training and validation curves  

The results highlight the **trade-off between performance and interpretability** in safety-critical NLP systems.

---

## 7. Repository Structure
```text
APPLIED-AI-NLP/
│
├── DATASET/
│   ├── RAW/                  # Original injury narratives
│   ├── PROCESSED/            # Cleaned and engineered datasets
│
├── Injury_Severity_Prediction.ipynb
│
├── *.png                     # Evaluation and explainability plots
├── nlp report.pdf            # Academic project report
├── nlp report.docx
├── .gitignore
└── README.md
8. Tools and Technologies
Python

scikit-learn

TensorFlow / Keras

NLTK / spaCy

Pandas, NumPy

Matplotlib, Seaborn

Jupyter Notebook

9. Academic Contribution
This project contributes to research on:

NLP for safety and risk analytics

Explainable AI in high-risk domains

Trustworthy and transparent machine learning

Human-in-the-loop decision support systems

The work aligns with current research themes in Explainable AI, applied NLP, and safety-critical machine learning.

10. PhD Research Extension
Potential PhD-level extensions include:

Post-hoc explainability methods (SHAP, LIME) for NLP models

Transformer-based architectures (BERT, RoBERTa) with explanation layers

Temporal modelling of injury narratives and severity trends

Multi-modal learning combining text with structured safety data

Deployment of explainable NLP systems in real industrial settings

11. Author
Anthony Eddei Kwofie
MSc Artificial Intelligence & Data Science
Explainable AI | NLP | Safety-Critical Systems

GitHub: https://github.com/Tony-Kwofie

12. License
This repository is intended for academic and research purposes.

yaml
Copy code

---

## Final Step: Commit the Update

```powershell
git add README.md
git commit -m "Update README with PhD-focused NLP and Explainable AI positioning"
git push
