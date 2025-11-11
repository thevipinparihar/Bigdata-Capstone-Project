🩺 README.md — Using Big Data Analytics to Diagnose Breast Cancer

Author: Vipin Kumar | Roll No: 24MBMB13
Program: MBA (Business Analytics), University of Hyderabad
Platform: Databricks • PySpark • scikit-learn • Plotly • NLTK

🧭 Project Overview

A comprehensive capstone integrating Machine Learning, Clustering, Feature Importance, Recurrence Prediction, and Text Analytics to uncover patterns in breast-cancer data for early detection and prognosis.

Dataset: workspace.default.breast_cancer_big_data_dataset
Structured + Unstructured (text) variables.

🧠 Case 1 – Early Detection of Breast Cancer (Structured Data)

Goal: Classify tumors as Benign (0) or Malignant (1).
Models: Logistic Regression | Random Forest

Metric	Logistic Regression	Random Forest
Accuracy	0.565	0.580
ROC-AUC	0.484	0.530
Precision (1)	0.42	0.48
Recall (1)	0.17	0.27
F1 (1)	0.24	0.34

Confusion Matrices LR [[99,19],[68,14]] RF [[94,24],[60,22]]
Insight: RF outperformed LR slightly; imbalance lowered malignant recall.
Conclusion: Established a baseline for later ensemble optimization.

🌲 Case 2 – Feature Importance Analysis for Diagnosis

Model: Random Forest Classifier | ROC-AUC = 0.5186

Top Features

Tumor Size (mm) 2. Mean Radius 3. Mean Smoothness 4. Compactness 5. Concavity
Observation: Though predictive power was limited, importance ranking identified clinically significant variables.
Conclusion: Tumor morphology metrics are primary diagnostic drivers.

🌀 Case 3 – Pattern Discovery through Clustering

Method: K-Means (k = 3) | Silhouette Score = 0.0906

Findings

Cluster 0: Benign pattern (small radius)

Cluster 1: Borderline cases

Cluster 2: Malignant pattern (large tumor, high compactness)

Visuals: 3D Scatter • Cluster Heatmap • Parallel Coordinates
Conclusion: Low silhouette indicates overlap, yet reveals progression continuum between benign and malignant groups.

⚡ Case 4 – Predicting Cancer Recurrence Risk

Models: Random Forest | Gradient Boosted Trees (GBT)

Model	ROC-AUC	Approx. Accuracy
Random Forest	0.5018	50 %
GBT	0.5071	51 %

Top Predictors: Tumor Size • Mean Radius • Genetic Risk Score
Insight: Marginal AUC shows limited signal in available variables; highlights need for richer clinical inputs.
Conclusion: Baseline recurrence models demonstrate methodology but require feature enhancement for clinical use.

🧾 Case 5 – Text Analytics on Pathology Reports

Goal: Mine unstructured reports for diagnostic clues using NLP.
Model: TF-IDF + Logistic Regression | Accuracy = 0.58 | ROC-AUC = 0.5831

Metric	Value
Precision (1)	0.51
Recall (1)	0.63
F1 (1)	0.56
Confusion Matrix	[[62 52],[32 54]]

Findings

Malignant reports → “invasive”, “carcinoma”, “grade III”

Benign reports → “fibroadenoma”, “no atypia”

LDA revealed five topics linked to cell grading and invasion.

Sentiment analysis showed negative tone for malignant cases.

Conclusion: Text-only signals (≈ 58 %) proved meaningful and enhance structured models when combined.

📊 Summary of Results
Case	Technique	Best Model	Key Metric
1	Classification	Random Forest	AUC 0.53
2	Feature Ranking	Random Forest	AUC 0.52
3	Clustering	K-Means (k=3)	Silhouette 0.09
4	Recurrence Prediction	GBT	AUC 0.51
5	NLP Text Model	Logistic Regression	AUC 0.58
🧩 Technologies Used

PySpark MLlib • scikit-learn • Matplotlib • Seaborn • Plotly • NLTK • Databricks • Python 3.

🎓 Conclusion & Future Scope

Integrated structured + unstructured analytics for diagnosis and prognosis.

Established baselines for ensemble and text models.

Future: Deep NLP (BERT), feature augmentation (genomics + imaging), deployment as clinical dashboard.

“Data-driven diagnosis can transform early detection and patient outcomes.”
