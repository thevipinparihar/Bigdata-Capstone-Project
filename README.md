# 🩺 Using Big Data Analytics to Diagnose Breast Cancer  
**Author:** Vipin Kumar | **Roll No:** 24MBMB13  
**Program:** MBA (Business Analytics), University of Hyderabad  
**Platform:** Databricks | PySpark | scikit-learn | Plotly | NLTK  

---

## 🧭 Project Overview
This capstone project applies **Big Data Analytics** to detect, predict, and analyze breast cancer using structured and unstructured data.  
It integrates **Machine Learning**, **Feature Importance**, **Clustering**, **Recurrence Prediction**, and **Text Analytics (NLP)** into a single end-to-end analytical workflow.

**Dataset:** `workspace.default.breast_cancer_big_data_dataset`  
Contains both structured attributes and textual pathology reports.

---

## 🧠 Case 1 — Early Detection of Breast Cancer (Supervised Learning)

### 🎯 Objective:
Classify tumors as **Benign (0)** or **Malignant (1)** using machine learning models.

### 🧰 Models Used:
- Logistic Regression  
- Random Forest Classifier  

### 📈 Evaluation Metrics:

| Metric | Logistic Regression | Random Forest |
|:--|:--:|:--:|
| Accuracy | 0.565 | **0.580** |
| ROC-AUC | 0.484 | **0.530** |
| Precision (1) | 0.42 | 0.48 |
| Recall (1) | 0.17 | 0.27 |
| F1 (1) | 0.24 | 0.34 |

**Confusion Matrices:**  
- Logistic Regression → `[[99,19],[68,14]]`  
- Random Forest → `[[94,24],[60,22]]`

### 💡 Insight:
Random Forest slightly outperformed Logistic Regression, improving malignant recall.  
Both models suffered from class imbalance but established a **baseline for further ensemble modeling**.

---

## 🌲 Case 2 — Feature Importance Analysis for Diagnosis

### 🎯 Objective:
Identify the most influential tumor features contributing to malignancy prediction.

### 🧰 Model Used:
Random Forest Classifier  

### 📈 Metric:
- **ROC-AUC:** 0.5186  

### 🔍 Top 10 Important Features:
1. Tumor_Size_mm  
2. Mean_Radius  
3. Mean_Smoothness  
4. Mean_Compactness  
5. Mean_Concavity  
6. Mean_Symmetry  
7. Genetic_Risk_Score  
8. Mean_Area  
9. Mean_Texture  
10. Mean_Fractal_Dimension  

### 💡 Insight:
Even with moderate predictive power, feature importance analysis revealed **clinically relevant biomarkers** such as tumor size and radius, key for diagnosis.

---

## 🌀 Case 3 — Pattern Discovery through Clustering

### 🎯 Objective:
Uncover hidden patterns and natural groupings of patients using unsupervised learning.

### 🧰 Model Used:
K-Means Clustering (k = 3)  

### 📈 Metric:
- **Silhouette Score:** 0.0906  

### 🔍 Observations:
- **Cluster 0:** Benign pattern (small radius, smooth edges)  
- **Cluster 1:** Borderline pattern (moderate tumor properties)  
- **Cluster 2:** Malignant pattern (large, irregular tumor features)  

### 💡 Insight:
Although clusters overlapped (low silhouette score), they revealed **transitional progression patterns** between benign and malignant cases.

---

## ⚡ Case 4 — Predicting Cancer Recurrence Risk

### 🎯 Objective:
Predict the **likelihood of cancer recurrence** using ensemble learning models.

### 🧰 Models Used:
- Random Forest Classifier  
- Gradient Boosted Trees (GBTClassifier)

### 📈 Evaluation Metrics:

| Model | ROC-AUC | Approx. Accuracy |
|:--|:--:|:--:|
| Random Forest | 0.5018 | 50% |
| Gradient Boosted Tree | **0.5071** | 51% |

### 🔍 Observations:
- Models show **limited predictive separation** (AUC ≈ 0.5).  
- **Top Predictors:** Tumor Size, Mean Radius, Genetic Risk Score.  
- Indicates missing or noisy recurrence-related features.

### 💡 Insight:
Recurrence prediction proved challenging with available data, demonstrating the **need for expanded variables** (e.g., treatment type, molecular data) to improve accuracy.

---

## 🧾 Case 5 — Text Analytics on Pathology Reports

### 🎯 Objective:
Extract diagnostic insights from **unstructured text data** (pathology reports) using NLP.

### 🧰 Methods:
- TF-IDF Vectorization + Logistic Regression  
- LDA Topic Modeling (Unsupervised)  
- VADER Sentiment Analysis  

### 📈 Evaluation Metrics:

| Metric | Value |
|:--|:--:|
| Accuracy | **0.58** |
| ROC-AUC | **0.5831** |
| Precision (1) | 0.51 |
| Recall (1) | 0.63 |
| F1 (1) | 0.56 |

**Confusion Matrix:** `[[62 52],[32 54]]`

### 🔍 Observations:
- Malignant reports contained words like **“invasive”**, **“carcinoma”**, **“grade III”**.  
- Benign reports included **“fibroadenoma”**, **“no atypia”**, **“benign tissue”**.  
- Topic modeling uncovered 5 themes related to invasion and grading.  
- Sentiment polarity: malignant = negative tone, benign = neutral tone.

### 💡 Insight:
Text-based models achieved **AUC ≈ 0.58**, revealing that unstructured data supports structured features for improved interpretability and diagnosis.

---

## 📊 Summary of Results

| Case | Technique | Best Model | Metric | Score |
|:--|:--|:--|:--|:--:|
| 1 | Classification | Random Forest | ROC-AUC | 0.53 |
| 2 | Feature Analysis | Random Forest | ROC-AUC | 0.52 |
| 3 | Clustering | K-Means | Silhouette | 0.09 |
| 4 | Recurrence Prediction | GBT | ROC-AUC | 0.51 |
| 5 | Text Analytics | Logistic Regression | ROC-AUC | 0.58 |

---

## 🧩 Tools & Technologies
| Category | Tools |
|:--|:--|
| Big Data Processing | Apache Spark (PySpark), Databricks |
| Machine Learning | scikit-learn, MLlib |
| Visualization | Plotly, Seaborn, Matplotlib |
| NLP & Text Mining | TF-IDF, LDA, NLTK, VADER |
| Language | Python 3.x |

---

## 🎓 Conclusion
- Integrated both **structured and unstructured analytics** to study breast cancer.  
- Achieved foundational prediction models with interpretable clinical insights.  
- Highlighted data gaps and need for advanced variables for recurrence forecasting.  
- NLP uncovered linguistic indicators that align with medical terminology.  

> “Data-driven healthcare analytics can empower early detection and support better clinical decisions.”

---

## 🚀 Future Scope
- Incorporate **deep learning (BERT/CNN)** for pathology text.  
- Combine genomic, image, and textual data for multimodal learning.  
- Deploy predictive API or dashboard for oncologists.

---

### 🏫 University of Hyderabad  
**MBA (Business Analytics) — Capstone Project (2025)**  
**Author:** Vipin Kumar | **Roll No:** 24MBMB13  

> *“Turning medical data into actionable insight through analytics.”*
