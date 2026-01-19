# Credit Risk Prediction System

An end-to-end Machine Learning project that predicts whether a loan applicant is a **High Risk** or **Low Risk** borrower using financial and demographic data. The system includes data preprocessing, model training, evaluation, and deployment using Streamlit.

---

## Objective
To reduce loan defaults by building a machine learning model that assists financial institutions in credit risk assessment.

---

##  Dataset
- Source: Kaggle – Credit Risk Dataset  
- Link: https://www.kaggle.com/datasets/laotse/credit-risk-dataset  
- Target Variable: `loan_status` (1 = High Risk, 0 = Low Risk)

---

##  Preprocessing & Feature Engineering
- Missing value handling using median imputation
- One-hot encoding for categorical features
- Feature scaling using StandardScaler
- Feature engineering:
  - Debt-to-Income Ratio
  - Total Obligation Ratio
  - Employment Length Grouping
- Class imbalance handled using SMOTE

---

##  Models Used
- Logistic Regression *(Final Model)*
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

---

## Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Logistic Regression was selected for deployment due to better interpretability and stable recall.

---

## Deployment
- Deployed using **Streamlit**
- User inputs applicant details via web interface
- Model returns risk prediction with probability score

Run locally:
```bash
streamlit run app.py
