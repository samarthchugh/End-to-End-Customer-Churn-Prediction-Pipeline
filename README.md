# 📉 End-to-End Customer Churn Prediction Pipeline
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/samarthchugh/telecom-churn-prediction-eda-model-development)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=python&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EB5E28)](https://xgboost.readthedocs.io/)

## 📌 Project Overview
Customer churn is a critical challenge for subscription-based businesses, where retaining existing customers is often more cost-effective than acquiring new ones. This project builds an end-to-end machine learning pipeline to predict customer churn and identify customers at risk of leaving, enabling businesses to take timely and targeted retention actions.

The project covers the full lifecycle of a data science solution — from exploratory data analysis (EDA) and feature understanding to model training, evaluation, and deployment-ready pipelines.

---

## 🎯 Business Objective
- Identify customers who are likely to churn
- Prioritize churn-risk customers for retention campaigns
- Minimize unnecessary retention costs caused by false alarms

Since churned customers form a minority class, the project focuses on recall and precision, rather than accuracy alone.

---

## 🧠 Key Concepts Used
- Class imbalance handling
- Precision–recall trade-off analysis
- Pipeline-based preprocessing and modeling
- Business-driven model selection
- Reproducible and production-ready ML design

---

## 📂 Project Structure
```bash
customer-churn-pipeline/
│
├── notebooks/
│   └── telecom_churn_eda_and_modeling.ipynb
│
├── src/
│   └── pipeline/
│       ├── preprocessing.py
│       ├── modeling.py
│       ├── train.py
│       └──evaluation.py
│
├── artifacts/
│   └── xgboost_churn_model.pkl
│
├── README.md
└── requirements.txt
```

---

## 🔍 Exploratory Data Analysis (EDA)
EDA was conducted in a Jupyter/Kaggle notebook to understand customer behavior, uncover churn patterns, and ensure that modeling decisions were driven by both data characteristics and business context.

Univariate analysis examined demographic features, tenure, contract type, and various service-related attributes to understand customer composition. The churn variable was analyzed independently, revealing a clear class imbalance with significantly fewer churned customers.
This insight directly motivated the use of recall-, precision-, and confusion-matrix–based evaluation rather than accuracy, which can be misleading in imbalanced datasets.

Bivariate analysis explored relationships between churn and key features such as tenure, contract type, senior citizen status, monthly charges, total charges, and internet service type. Additional analysis examined interactions between monthly charges and internet services to better understand pricing-related churn behavior.
These analyses revealed higher churn rates among short-tenure customers, month-to-month contracts, and customers with higher monthly charges.

Churn rates were further analyzed across customer segments defined by tenure groups, contract types, seniority, and charge ranges. These insights highlighted the importance of minimizing missed churners, while also recognizing the cost implications of excessive false positives.

Beyond descriptive analysis, model-oriented EDA was performed. Feature importance from a Random Forest model was used to gain an initial understanding of influential attributes without manual feature elimination. ROC–AUC curves were plotted for SVM, Random Forest, and XGBoost to compare their discriminative ability. Precision–recall comparisons across all models were used to explicitly evaluate trade-offs between churn detection and false churn alerts.

Overall, EDA was used not for aggressive feature removal, but to inform metric selection, class imbalance handling, and model evaluation strategy, ensuring alignment with real-world churn prevention objectives. EDA was intentionally kept separate from the production pipeline, as its purpose was analytical insight rather than reusable preprocessing logic.

---

## ⚙️ Data Preprocessing
Key preprocessing steps include:
- Conversion of `TotalCharges` to numeric values
- Handling missing values
- Encoding categorical variables using `OneHotEncoder`
- Mapping churn labels (`Yes → 1`, `No → 0`)
- Removing irrelevant identifiers (e.g., `customerID`)

All preprocessing steps are implemented using a `ColumnTransformer` and integrated into the model pipeline to prevent data leakage and ensure consistent transformations during training and inference.

---

## 🤖 Models Evaluated
The following models were trained and evaluated:
- Logistic Regression (baseline)
- Random Forest (balanced)
- Support Vector Machine (balanced)
- AdaBoost
- XGBoost

Model evaluation focused on:
- **Recall (Churn = Yes)** – ability to identify churned customers
- **Precision (Churn = Yes)** – reliability of churn predictions
- Confusion matrix analysis for error inspection

---

## 📊 Model Performance & Comparison
- Models such as SVM and Random Forest achieved higher recall but produced a large number of false positives, increasing potential retention costs.
- AdaBoost achieved higher precision but missed a significant portion of churned customers.
- XGBoost demonstrated the best balance between recall and precision, offering a more practical trade-off for real-world churn prevention.

Precision–recall comparison charts and confusion matrices were used to visually interpret these trade-offs and support model selection decisions.

---

## ✅ Final Model Selection: XGBoost
XGBoost was selected as the final model due to its balanced and stable performance. While it does not maximize recall, it significantly reduces false positives and provides a better precision–recall trade-off.

This balance is critical in real-world scenarios, where retention efforts must be both effective and cost-efficient. XGBoost enables targeted intervention for high-risk customers without overspending on unnecessary retention actions.

---

## 🧪 Evaluation Strategy
- Precision, recall, and F1-score focused on churn class
- Confusion matrix analysis for business impact interpretation
- ROC–AUC curves for model discrimination comparison
- Precision–recall trade-off analysis for cost-sensitive decision-making

Accuracy alone was intentionally avoided as a primary metric due to class imbalance.

---

## 🚀 Training Pipeline
The training pipeline:
1. Loads raw customer data
2. Performs data cleaning and preprocessing
3. Splits data into training and test sets
4. Trains an XGBoost model within a pipeline
5. Handles class imbalance using scale_pos_weight
6. Saves the trained model for inference

All components are modular, reproducible, and production-oriented.

---

## 🧠 Key Takeaways
- Churn prediction is a cost-sensitive, imbalanced classification problem
- High recall alone is insufficient for business decisions
- Precision–recall trade-offs are essential for model selection
- Pipeline-based design prevents data leakage
- Business context should guide evaluation and deployment choices

---

## 📌 Final Recommendation
XGBoost is recommended as the final model due to its balanced performance, stability, and practical usability. It enables businesses to focus on high-risk customers, reduce unnecessary retention costs, and implement effective churn prevention strategies.

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Joblib

---

## 👤 Author
[Samarth Chugh](https://www.linkedin.com/in/-samarthchugh/)\
[Kaggle](https://www.kaggle.com/code/samarthchugh/telecom-churn-prediction-eda-model-development)\
Aspiring Data Scientist / Machine Learning Engineer\
Focused on building business-driven, production-ready ML solutions.

