# 🩺 Using Big Data Analytics to Diagnose Breast Cancer
### 📊 Capstone Project — MBA (Business Analytics) | University of Hyderabad  
**Author:** Vipin Kumar (24MBMB13)  
**Tools Used:** Databricks | PySpark | Python | Scikit-learn | Plotly | Seaborn | NLP | MLlib  

---

## 🧭 Project Overview
This capstone project applies **Big Data Analytics** techniques to the healthcare domain to detect, predict, and analyze **breast cancer** using both structured and unstructured data.  
The analysis integrates **machine learning**, **clustering**, **feature importance**, and **text analytics** to uncover clinical insights.

---

## 🧩 Dataset Description
Dataset used: `workspace.default.breast_cancer_big_data_dataset` (loaded in Databricks)

| Column Type | Description |
|--------------|--------------|
| Numeric | Tumor size, Mean radius, Texture, Smoothness, Compactness, etc. |
| Categorical | Diagnosis (Benign / Malignant), Hormone receptor status, HER2 status |
| Text | Pathology report descriptions |
| Target Variables | Diagnosis, Recurrence Risk |

---

# 🧠 Case 1: Early Detection of Breast Cancer Using Machine Learning

### 🎯 Objective:
Predict whether a tumor is **benign or malignant** using supervised learning models.

### 🧰 Methods:
- Logistic Regression  
- Random Forest Classifier  
- PySpark + Scikit-learn hybrid workflow  
- Feature scaling and evaluation with ROC-AUC  

### 📈 Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score  
- ROC Curve & AUC  
- Confusion Matrix

### 🎨 Visualizations:
- ROC Curves (LR vs RF)  
- Feature correlation heatmap  
- Confusion matrices  
- Feature importance bar chart  
- Model comparison bar chart  

### 📊 Key Insight:
> Random Forest achieved higher accuracy and robustness (AUC ≈ 0.58).  
> Tumor radius, smoothness, and compactness were key predictive features.

---

# 🔍 Case 2: Feature Importance Analysis for Diagnosis

### 🎯 Objective:
Identify the most influential features for cancer diagnosis using **Random Forest** in PySpark.

### 🧰 Techniques:
- Feature vector creation using VectorAssembler  
- Model training via `RandomForestClassifier`  
- Feature ranking and correlation analysis  

### 📈 Evaluation Metric:
- ROC-AUC: ~0.93 (Excellent predictive power)

### 🎨 Visualizations:
- 3D interactive bar chart of feature importances  
- Correlation heatmap  
- Parallel coordinates of top features  
- 3D importance surface  

### 📊 Key Insight:
> Top predictors: Tumor Size, Mean Radius, and Smoothness.  
> These features are clinically associated with malignancy severity.

---

# 🌀 Case 3: Pattern Discovery through Clustering

### 🎯 Objective:
Discover **hidden patterns** in patient data using unsupervised learning (K-Means Clustering).

### 🧰 Methods:
- Feature scaling and K-Means clustering (k=3)  
- Silhouette Score for evaluation  
- 2D and 3D cluster visualizations  

### 📈 Evaluation Metric:
- Silhouette Score: ~0.71 (Strong separation)

### 🎨 Visualizations:
- 3D interactive cluster plot  
- Cluster surface landscape  
- Cluster size distribution pie chart  
- Parallel coordinates plot  
- Cluster centroid heatmap  

### 📊 Key Insight:
> Data naturally grouped into 3 clusters resembling *benign*, *borderline*, and *malignant* groups.  
> Tumor radius and compactness strongly differentiate clusters.

---

# ⚡ Case 4: Predicting Cancer Recurrence Risk

### 🎯 Objective:
Predict the **recurrence risk** (high or low) using ensemble models.

### 🧰 Methods:
- Random Forest Classifier  
- Gradient Boosted Trees (GBTClassifier)  
- Feature scaling and evaluation  

### 📈 Evaluation Metrics:
| Model | AUC Score |
|--------|------------|
| Random Forest | 0.92 |
| Gradient Boosted Tree | **0.95** |

### 🎨 Visualizations:
- AUC comparison bar chart  
- ROC curve (RF vs GBT)  
- 3D feature importance plot  
- Risk distribution histogram  
- 3D risk probability surface  
- Correlation heatmap  

### 📊 Key Insight:
> Gradient-Boosted Trees outperformed RF (AUC = 0.95).  
> Tumor Size, Mean Radius, and Genetic Risk Score were top predictors for recurrence.

---

# 🧾 Case 5: Text Analytics on Pathology Reports

### 🎯 Objective:
Use **Natural Language Processing (NLP)** to extract patterns and clinical themes from pathology reports.

### 🧰 Techniques:
- Text cleaning & preprocessing  
- TF-IDF vectorization  
- Topic Modeling (LDA)  
- Logistic Regression text classifier  
- Sentiment analysis (VADER)  

### 📈 Evaluation:
- Accuracy: ~85%  
- ROC-AUC: ~0.88  
- Topics: “Invasive Carcinoma”, “Ductal Patterns”, “Cell Grading”, etc.

### 🎨 Visualizations:
| Visual | Description |
|--------|--------------|
| 🧠 Word Clouds | Malignant vs Benign terms |
| 📊 Word Frequency Bar Chart | Comparative linguistic usage |
| 🔮 3D Topic Distribution | Interactive clustering of text |
| 🌀 t-SNE Topic Projection | 2D reduction for document separation |
| 💬 Sentiment Histogram | Emotional tone of reports |
| 🔗 Topic-Term Network | Interactive graph linking topics to terms |

### 📊 Key Insight:
> Malignant reports show terms like “invasive”, “carcinoma”, and “grade III” —  
> whereas benign reports include “fibroadenoma” and “no atypia”.  
> Text-based predictions support structured-data models for improved diagnostic accuracy.

---

# 🧮 Technology Stack

| Component | Technology |
|------------|-------------|
| Big Data Processing | Apache Spark (PySpark) |
| Machine Learning | MLlib, scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Text Analytics | TF-IDF, LDA, NLTK |
| Platform | Databricks |
| Programming Language | Python 3.x |

---

# 📊 Evaluation Summary

| Case | Method | Model(s) | Metric | Result |
|-------|----------|-----------|----------|----------|
| Case 1 | Classification | Logistic Regression / RF | AUC | 0.58 |
| Case 2 | Feature Ranking | Random Forest | AUC | 0.93 |
| Case 3 | Clustering | K-Means | Silhouette | 0.71 |
| Case 4 | Prediction | RF / GBT | AUC | **0.95** |
| Case 5 | NLP | Logistic Regression | Accuracy | 85% |

---

# 📁 Deliverables
1. **Project Demo Video:** 10–15 minutes walkthrough  
2. **Presentation File:** PowerPoint summarizing all cases  
3. **Project Repository:**  
   - `.py` or `.ipynb` code files (per case)  
   - `.csv` datasets and outputs (saved in FileStore)  
   - `.pdf` report or PPT  
   - This `README.md`  

---

# 🧠 Insights and Conclusion
- The project successfully integrates **structured data analytics** and **text analytics** to diagnose and analyze breast cancer.  
- The **GBT model** achieved excellent accuracy in recurrence risk prediction.  
- **NLP analysis** revealed distinct medical terminologies in malignant vs benign reports.  
- Combining **clinical features** with **pathology text data** can enhance predictive accuracy and aid early detection.

---

# 🏆 Future Enhancements
- Deploy model as a **web API or dashboard** for oncologists  
- Implement **BERT-based deep NLP models** for advanced text understanding  
- Integrate **real-time data streaming** via Apache Kafka  

---

# 💻 Author

**👩‍💼 Vipin Kumar(24MBMB13)**  
MBA (Business Analytics) — University of Hyderabad  
**Roles:** Project Lead, Data Scientist, Business Analyst  
**Focus:** Healthcare analytics, big data modeling, ethical AI  

---

# 📎 Repository Structure

