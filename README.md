# Explainable NLP for Injury Severity Prediction  
**MSc Artificial Intelligence & Data Science | University of Hull (London Campus)**

---

## 1. Research Motivation
Occupational injuries remain a major challenge in safety-critical industries such as construction, mining, and manufacturing. While large volumes of incident reports are routinely collected, much of this data is unstructured and under-utilised.

This project investigates how **Natural Language Processing (NLP)** and **Explainable Artificial Intelligence (XAI)** can be used to transform free-text injury narratives into **transparent, interpretable, and actionable safety intelligence**. The work is motivated by the need for **trustworthy AI systems** in high-risk environments where model decisions must be clearly understood by safety professionals.

---

## 2. Research Objectives
- Develop NLP models to predict injury severity from occupational incident narratives  
- Compare classical machine learning approaches with deep learning sequence models  
- Address real-world data challenges including noise, imbalance, and missing values  
- Integrate explainability techniques to support transparent decision-making  
- Explore the role of XAI in safety-critical AI systems  

---

## 3. Methodology

### 3.1 Data and Preprocessing
- Real-world occupational injury narratives (OSHA-style reports)  
- Text cleaning, normalisation, and tokenisation  
- Narrative length analysis and exploratory text statistics  
- Handling class imbalance using SMOTE  

### 3.2 Machine Learning Models
- TF-IDF feature representations  
- Logistic Regression with resampling  
- Performance evaluated using:
  - Precision–Recall curves  
  - ROC–AUC  
  - F1-score  
  - Confusion matrices  

### 3.3 Deep Learning Models
- Bidirectional Long Short-Term Memory (BiLSTM)  
- Sequential modelling for contextual understanding  
- Analysis of training dynamics and convergence behaviour  

---

## 4. Explainable AI (XAI) Perspective
A core contribution of this work is its emphasis on **interpretability in NLP-based safety models**. Rather than treating injury severity prediction as a black-box problem, the project focuses on:

- Identification of influential words and phrases associated with severe injuries  
- Class-specific keyword analysis and word clouds  
- Feature importance interpretation for linear models  
- Comparative analysis between interpretable classical models and deep neural networks  

This approach supports **human-centred AI**, enabling safety managers and regulators to:
- Understand *why* a prediction was made  
- Identify risk-driving linguistic patterns  
- Build trust in AI-assisted safety decisions  

---

## 5. Experimental Results
The repository includes extensive quantitative and visual evaluation:

- Confusion matrices and classification reports  
- Precision–Recall and ROC curves  
- F1-score comparisons across models  
- Narrative length distribution analysis  
- Word clouds for hospitalized vs non-hospitalized cases  
- BiLSTM training and validation curves  

The results highlight the **trade-off between predictive performance and interpretability** in safety-critical NLP systems.

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

The project was developed using the following tools and libraries:

Programming Language: Python

Machine Learning: scikit-learn

Deep Learning: TensorFlow / Keras

Natural Language Processing: NLTK, spaCy

Data Analysis: Pandas, NumPy

Visualisation: Matplotlib, Seaborn

Development Environment: Jupyter Notebook

8. Academic Contribution

This work contributes to ongoing research in:

NLP-based safety and risk analytics

Explainable Artificial Intelligence (XAI) in high-risk and regulated domains

Trustworthy and transparent machine learning systems

Human-in-the-loop decision support for safety management

The project aligns with contemporary research themes in Explainable AI, applied NLP, and safety-critical machine learning.

9. PhD Research Extension

Potential PhD-level research extensions include:

Post-hoc explainability methods (e.g. LIME, SHAP) for NLP models

Transformer-based architectures (e.g. BERT, RoBERTa) with explanation layers

Temporal modelling of injury narratives and severity trends

Multi-modal learning combining text with structured safety data

Deployment of explainable NLP systems in real industrial environments

10. Author

Anthony Eddei Kwofie
MSc Artificial Intelligence & Data Science

Research Interests:
Explainable AI (XAI), NLP, Safety-Critical Systems, Occupational Health & Safety

GitHub: https://github.com/Tony-Kwofie

LinkedIn: (add link if desired)

11. License

This repository is intended for academic and research purposes only.
