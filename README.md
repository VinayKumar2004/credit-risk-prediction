# 🏦 Credit Risk Prediction for Loan Applicants

## 📌 Problem Statement

Financial institutions face significant challenges in assessing the creditworthiness of loan applicants. Accurate credit risk prediction is essential for minimizing defaults and maintaining the stability of the lending ecosystem. This project leverages the **German Credit Dataset** to develop a predictive machine learning model that classifies applicants as either **Good Credit Risk** or **Bad Credit Risk**.

---

## 🎯 Objective

- Build a predictive model to classify credit risk using applicant financial and demographic data.
- Engineer and evaluate new features to improve model performance.
- Interpret model outputs to uncover insights into creditworthiness.
- Present findings through a **Streamlit dashboard** for intuitive interaction and explainability.

---

## 📊 Key Features

- **Exploratory Data Analysis (EDA)** with visual summaries of data distribution and relationships.
- **Feature Engineering**, including:
  - `Expected_Loss` based on EAD, PD, and LGD principles.
  - `Credit_per_Age`, `Duration_per_Amount`, and domain-specific interaction terms.
- **Modeling Approaches**:
  - Logistic Regression
  - Random Forest (with feature importance)
  - XGBoost (with feature importance)
- **Hyperparameter tuning** via `GridSearchCV`.
- **Model evaluation** using accuracy, precision, recall, and F1-score.
- **Interactive user input** and real-time prediction for custom applicant profiles.
- **Visualized model insights** and importance of engineered features.
- **Business recommendations** based on model interpretation.

---

## 🛠️ Tech Stack

- **Python** with libraries:
  - `pandas`, `numpy` for data processing
  - `scikit-learn`, `xgboost` for machine learning
  - `matplotlib`, `seaborn` for visualizations
  - `streamlit` for the interactive web app

---

## 📁 Project Structure

📦Credit-Risk-Prediction  
┣ 📄 main.py ← Streamlit app and core logic  
┣ 📄 german_credit_data.csv ← Dataset used  
┣ 📄 requirements.txt ← List of dependencies   
┗ 📄 README.md ← Project overview

---

## 🚀 How to Run

1. Clone the repository and navigate to the project folder.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit dashboard:
   ```bash
    streamlit run main.py
  ```
