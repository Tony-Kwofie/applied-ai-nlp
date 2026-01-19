# Explainable NLP for Injury Severity Prediction  
**MSc Artificial Intelligence & Data Science | University of Hull (London Campus)**

## 1. Research Motivation
Occupational injuries remain a major challenge in safety-critical industries such as construction, mining, and manufacturing. While large volumes of incident reports are routinely collected, much of this data is unstructured and under-utilised.

This project investigates how **Natural Language Processing (NLP)** and **Explainable Artificial Intelligence (XAI)** can be used to transform free-text injury narratives into **transparent, interpretable, and actionable safety intelligence**. The work is motivated by the need for **trustworthy AI systems** in high-risk environments where model decisions must be understood by safety professionals.

---

## 2. Research Objectives
- Develop NLP models to predict injury severity from occupational incident narratives  
- Compare classical machine learning approaches with deep learning sequence models  
- Address real-world data challenges including noise, imbalance, and missing values  
- Integrate **explainability techniques** to support transparent decision-making  
- Explore the role of XAI in safety-critical AI systems  

---

## 3. Methodology

### 3.1 Data and Preprocessing
- Real-world occupational injury narratives (OSHA-style reports)
- Text cleaning, normalization, and tokenization
- Narrative length analysis and exploratory text statistics
- Handling class imbalance using SMOTE

### 3.2 Machine Learning Models
- TF-IDF feature representations
- Logistic Regression with resampling
- Performance evaluated using:
  - Precision–Recall curves
  - ROC–AUC
  - F1-Score
  - Confusion Matrices

### 3.3 Deep Learning Models
- Bidirectional Long Short-Term Memory (BiLSTM)
- Sequence modelling for contextual understanding
- Analysis of training dynamics and convergence behaviour

---

## 4. Explainable AI (XAI) Perspective
A core contribution of this work is the focus on **interpretability in NLP-based safety models**. Rather than treating injury severity prediction as a black-box problem, the project emphasises:

- Identification of influential words and phrases associated with severe injuries  
- Class-specific keyword analysis and word clouds  
- Feature importance interpretation for linear models  
- Comparison between interpretable classical models and deep neural networks  

This approach supports **human-centred AI**, enabling safety managers and regulators to:
- Understand *why* a prediction was made  
- Identify risk-driving linguistic patterns  
- Trust AI-assisted safety decisions  

---

## 5. Experimental Results
The repository includes extensive visual and quantitative evaluation:

- Confusion matrices and classification reports
- Precision–Recall and ROC curves
- F1-score comparisons across models
- Narrative length distributions
- Word clouds for hospitalized vs non-hospitalized cases
- BiLSTM training and validation curves

These results demonstrate the trade-off between predictive performance and interpretability in safety-critical NLP systems.

---

## 6. Repository Structure
```text
APPLIED-AI-NLP/
│
├── DATASET/
│   ├── RAW/                  # Original incident narratives
│   ├── PROCESSED/            # Cleaned and engineered datasets
│
├── Injury_Severity_Prediction.ipynb
│
├── *.png                     # Evaluation and explainability plots
├── nlp report.pdf            # Academic project report
├── nlp report.docx
├── .gitignore
└── README.md
7. Tools and Technologies
Python

scikit-learn

TensorFlow / Keras

NLTK / spaCy

Pandas, NumPy

Matplotlib, Seaborn

Jupyter Notebook

8. Academic Contribution
This project contributes to ongoing research on:

NLP for safety analytics

Explainable AI in high-risk domains

Trustworthy and transparent machine learning

Human-in-the-loop decision support systems

The work aligns with research themes in XAI, applied machine learning, and AI for societal and industrial impact.

9. PhD Research Extension
Potential PhD-level extensions include:

Post-hoc explainability methods (LIME, SHAP) for NLP models

Transformer-based architectures (BERT, RoBERTa) with explanation layers

Temporal modelling of injury narratives and severity trends

Multi-modal learning combining text with structured safety data

Deployment of explainable NLP models in real industrial settings

10. Author
Anthony Eddei Kwofie
MSc Artificial Intelligence & Data Science
Occupational Health & Safety | Explainable AI | NLP

GitHub: https://github.com/Tony-Kwofie
LinkedIn: (add link if desired)

11. License
This repository is intended for academic and research purposes.
