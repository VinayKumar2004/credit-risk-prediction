# Credit Risk Prediction using Expected Loss Framework and Machine Learning - Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("📊 Credit Risk Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv")
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')
    return df

def calculate_expected_loss(row):
    ead = row['Credit amount']
    pd_ = 0.05
    if row['Job'] == 0: pd_ += 0.05
    elif row['Job'] == 1: pd_ += 0.03
    elif row['Job'] == 2: pd_ += 0.02
    elif row['Job'] == 3: pd_ += 0.01

    if row['Saving accounts'] == 'little': pd_ += 0.04
    elif row['Saving accounts'] == 'moderate': pd_ += 0.02
    elif row['Saving accounts'] in ['quite rich', 'rich']: pd_ -= 0.01

    if row['Checking account'] == 'little': pd_ += 0.03
    elif row['Checking account'] == 'moderate': pd_ += 0.02
    elif row['Checking account'] == 'rich': pd_ -= 0.01

    if row['Purpose'] in ['business', 'education']: pd_ += 0.02
    elif row['Purpose'] in ['car', 'furniture/equipment']: pd_ += 0.01

    pd_ = min(max(pd_, 0.01), 0.5)
    lgd = 0.5
    if row['Housing'] == 'own': lgd -= 0.1
    elif row['Housing'] == 'free': lgd += 0.1
    if row['Duration'] > 36: lgd += 0.05
    elif row['Duration'] < 12: lgd -= 0.05
    lgd = min(max(lgd, 0.1), 0.9)

    return ead * pd_ * lgd

def preprocess(df):
    df['Expected_Loss'] = df.apply(calculate_expected_loss, axis=1)
    df['Credit_per_Age'] = df['Credit amount'] / df['Age']
    df['Duration_per_Amount'] = df['Duration'] / df['Credit amount']
    median_el = df['Expected_Loss'].median()
    df['Credit_Risk'] = df['Expected_Loss'].apply(lambda x: 'Bad' if x > median_el else 'Good')

    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    X = df_encoded.drop(columns=['Credit_Risk', 'Expected_Loss'])
    y = df_encoded['Credit_Risk'].map({'Good': 0, 'Bad': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, df_encoded

def train_models(X_train, y_train):
    param_grid_rf = {'n_estimators': [100], 'max_depth': [None]}
    param_grid_xgb = {'n_estimators': [100], 'max_depth': [3], 'learning_rate': [0.1]}

    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='f1')
    grid_rf.fit(X_train, y_train)

    grid_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid_xgb, cv=5,scoring='f1')
    grid_xgb.fit(X_train, y_train)

    models = {
        'Random Forest': grid_rf.best_estimator_,
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': grid_xgb.best_estimator_
    }

    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred)
        }
    return results

# Load and preprocess
st.sidebar.header("Settings")
raw_df = load_data()
X_train, X_test, y_train, y_test, df_encoded = preprocess(raw_df)

# Exploratory Data Analysis
if st.sidebar.checkbox("Show Exploratory Data Analysis"):
    st.subheader("📊 Exploratory Data Analysis")
    st.write("Target Variable Distribution:")
    st.bar_chart(raw_df['Credit_Risk'].value_counts())

    st.write("Credit Amount Distribution by Risk")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Credit_Risk', y='Credit amount', data=df_encoded, ax=ax3)
    st.pyplot(fig3)

# User Input Section
if st.sidebar.checkbox("Try a Prediction (Manual Input)"):
    st.subheader("🔎 Predict Credit Risk from User Input")
    user_input = {}
    user_input['Duration'] = st.slider("Duration (months)", 4, 72, 24)
    user_input['Credit amount'] = st.slider("Credit Amount", 250, 20000, 5000)
    user_input['Age'] = st.slider("Age", 18, 75, 35)
    user_input['Job'] = st.selectbox("Job", [0, 1, 2, 3])
    user_input['Sex'] = st.selectbox("Sex", ['male', 'female'])
    user_input['Housing'] = st.selectbox("Housing", ['own', 'free', 'rent'])
    user_input['Saving accounts'] = st.selectbox("Saving Accounts", ['little', 'moderate', 'quite rich', 'rich', 'unknown'])
    user_input['Checking account'] = st.selectbox("Checking Account", ['little', 'moderate', 'rich', 'unknown'])
    user_input['Purpose'] = st.selectbox("Purpose", ['car', 'furniture/equipment', 'radio/TV', 'education', 'business', 'domestic appliances', 'repairs', 'vacation/others'])

    input_df = pd.DataFrame([user_input])
    input_df['Expected_Loss'] = input_df.apply(calculate_expected_loss, axis=1)
    median_el = raw_df['Credit amount'].apply(lambda x: calculate_expected_loss(pd.Series({**user_input, 'Credit amount': x}))).median()
    prediction = 'Bad' if input_df['Expected_Loss'].iloc[0] > median_el else 'Good'
    st.write(f"### 🧾 Predicted Credit Risk: **{prediction}**")

# Train models and evaluate
models = train_models(X_train, y_train)
results = evaluate_models(models, X_test, y_test)

st.subheader("📈 Model Performance")
results_df = pd.DataFrame(results).T
st.dataframe(results_df.style.highlight_max(axis=0))

fig, ax = plt.subplots(figsize=(10, 5))
results_df.plot(kind='bar', ax=ax)
st.pyplot(fig)

# Feature Importance
st.subheader("🔍 Feature Importance (Random Forest)")
importances = models['Random Forest'].feature_importances_
feature_names = df_encoded.drop(columns=['Credit_Risk', 'Expected_Loss']).columns
imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
fig2, ax2 = plt.subplots()
sns.barplot(x=imp_series.values, y=imp_series.index, ax=ax2)
ax2.set_title("Feature Importance")
st.pyplot(fig2)

xgb_imp = pd.Series(models['XGBoost'].feature_importances_, index=feature_names).sort_values(ascending=False)
fig_xgb, ax_xgb = plt.subplots()
sns.barplot(x=xgb_imp.values, y=xgb_imp.index, ax=ax_xgb, palette="rocket")
ax_xgb.set_title("Feature Importance - XGBoost")
st.pyplot(fig_xgb)

# Logistic Regression
models['Logistic Regression'].fit(X_train, y_train)
logreg_imp = pd.Series(np.abs(models['Logistic Regression'].coef_[0]), index=feature_names).sort_values(ascending=False)
fig_lr, ax_lr = plt.subplots()
sns.barplot(x=logreg_imp.values, y=logreg_imp.index, ax=ax_lr, palette="viridis")
ax_lr.set_title("Feature Importance - Logistic Regression")
st.pyplot(fig_lr)

engineered_features = ['Credit_per_Age', 'Duration_per_Amount', 'Credit_Amount_per_Job', 'Age_Duration_Interaction']
engineered_imp = imp_series[imp_series.index.isin(engineered_features)]

st.subheader("🧪 Engineered Feature Importance")
fig_eng, ax_eng = plt.subplots()
sns.barplot(x=engineered_imp.values, y=engineered_imp.index, ax=ax_eng, palette='crest')
ax_eng.set_title("Engineered Features Impact")
st.pyplot(fig_eng)

# Recommendations
st.subheader("💡 Recommendations")
st.markdown("""
- Focus credit policy decisions on key features like **Duration**, **Credit Amount**, **Credit_per_Age**, and **Checking Account** status.
- Applicants with low checking and saving account scores should undergo stricter financial scrutiny.
- Consider using new features like **Credit_per_Age** and **Duration_per_Amount** to better assess affordability.
- Use XGBoost as a preferred model due to its robustness and high performance on imbalanced data.
- Integrate additional behavioral data (e.g., past payment trends, digital financial behavior) for more robust models.
""")
