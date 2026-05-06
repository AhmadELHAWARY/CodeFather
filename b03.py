import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

education_map = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}

st.set_page_config(page_title="CodeFather", page_icon="💷", layout="centered")
st.title("CodeFather")
st.markdown("Let's predict your salary 🤩")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_models():
    model_names = ["Logistic_Regression", "Decision_Tree",
                   "Random_Forest", "KNN", "SVM"]
    models = {}
    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        models[name.replace("_", " ")] = joblib.load(path)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    return models, scaler, le

trained_models, scaler, le = load_models()

st.sidebar.header("⚙️ Settings")
chosen_name = st.sidebar.selectbox("Choose a model", list(trained_models.keys()))

st.subheader("📋 Enter Person Details")
col1, col2 = st.columns(2)

with col1:
    age            = st.number_input("Age", min_value=17, max_value=90, value=30)
    #education_num  = st.slider("Education Level (1–16)", 1, 16, 10)
    capital_gain   = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss   = st.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)

with col2:
    education = st.selectbox("education", ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
                                           'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'])

    marital_status = st.selectbox("Marital Status", [
        "Never-married", "Married-civ-spouse", "Divorced",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])
    if marital_status in ["Married-civ-spouse", "Married-AF-spouse"]:
        relationship = st.selectbox("Relationship", [
            "Wife",  "Husband"
        ])
    elif marital_status == "Never-married":
        relationship = st.selectbox("Relationship", ["Own-child", "Unmarried", "Other-relative"])
    else:
        relationship = st.selectbox("Relationship", ["Unmarried", "Own-child", "Other-relative", "Not-in-family"])

    position = st.selectbox("Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ])


# Must match CodeFather.py exactly
onehot_cols   = ["marital-status", "position", "relationship"]
cols_to_scale = ['capital-gain', 'capital-loss', 'age', 'hours-per-week']  # 4 cols, MinMaxScaler

X_train_cols = trained_models[chosen_name].feature_names_in_
education_num=education_map.get(education)
if st.button("🔮 Predict Salary", use_container_width=True):
    user_df = pd.DataFrame([{
        'age':            age,
        'education-num':  education_num,
        'capital-gain':   capital_gain,
        'capital-loss':   capital_loss,
        'hours-per-week': hours_per_week,
        'marital-status': marital_status,
        'position':       position,
        'relationship':   relationship
    }])

    # 1. Scale — same 4 cols, same MinMaxScaler as CodeFather.py
    user_df[cols_to_scale] = scaler.transform(user_df[cols_to_scale])

    # 2. One-hot encode
    for col in onehot_cols:
        dummies = pd.get_dummies(user_df[col], prefix=col).astype(int)
        user_df = user_df.drop(columns=[col])
        user_df = pd.concat([user_df, dummies], axis=1)

    # 3. Align to training columns
    user_encoded = user_df.reindex(columns=X_train_cols, fill_value=0)

    # 4. Predict
    model      = trained_models[chosen_name]
    prediction = model.predict(user_encoded)
    label      = le.inverse_transform(prediction)[0]

    if ">50K" in label:
        st.success(f"✅ Predicted Salary: **{label}**")
    else:
        st.warning(f"📊 Predicted Salary: **{label}**")

