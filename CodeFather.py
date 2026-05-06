import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import os
import joblib

# ─── Step 1: Load and Clean ───────────────────────────────────
df_train = pd.read_csv(r"C:\Users\ahmed\Downloads\Train.csv")
df_test = pd.read_csv(r"C:\Users\ahmed\Downloads\Test.csv")

education_map = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9,
    "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
}

# Standardize column names
df_test = df_test.rename(columns={
    "workclass": "work-class",
    "fnlwgt":    "work-fnl",
    "occupation":"position"
})

# Remove duplicates
df_train = df_train.drop_duplicates()
df_test  = df_test.drop_duplicates()

# Handle hidden nulls (' ?') then strip ALL string columns
df_train.replace(" ?", np.nan, inplace=True)
df_test.replace(" ?", np.nan, inplace=True)
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

# ✅ Strip leading/trailing spaces from every string column

for col in df_train.select_dtypes(include='object').columns:
    df_train[col] = df_train[col].str.strip()
for col in df_test.select_dtypes(include='object').columns:
    df_test[col] = df_test[col].str.strip()

# Encode target
target = "salary"
le = LabelEncoder()
df_train["salary"] = le.fit_transform(df_train["salary"])
df_test["salary"]  = le.transform(df_test["salary"])

#
# # Confirm mapping is correct
# print("🔢 Salary encoding:")
# for i, cls in enumerate(le.classes_):
#     print(f"   {cls} → {i}")


# Drop unused columns
drop_cols = ['work-fnl', 'education', 'sex', 'race', 'native-country', 'work-class']
df_train.drop(drop_cols, axis=1, inplace=True)
df_test.drop(drop_cols,  axis=1, inplace=True)
onehot_cols  = ["marital-status", "position", "relationship"]
cols_to_scale = ['capital-gain', 'capital-loss', 'age', 'hours-per-week']




# Scale numeric columns
scaler = MinMaxScaler()
df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
df_test[cols_to_scale]  = scaler.transform(df_test[cols_to_scale])



# One-hot encode categorical columns
train_size = len(df_train)
combined   = pd.concat([df_train, df_test], axis=0)
combined   = pd.get_dummies(combined, columns=onehot_cols).astype(int)
df_train   = combined.iloc[:train_size]
df_test    = combined.iloc[train_size:]

X_train = df_train.drop(columns=[target])
y_train = df_train[target]
X_test  = df_test.drop(columns=[target])
y_test  = df_test[target]




# ─── ✅ FIX: Apply SMOTE to balance the training set ─────────
print("⚖️  Balancing training data with SMOTE...")
print(f"   Before → {dict(zip(*np.unique(y_train, return_counts=True)))}")

sm = SMOTE(random_state=42, sampling_strategy=0.5)  # Balance to 50% minority class
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

print(f"   After  → {dict(zip(*np.unique(y_train_bal, return_counts=True)))}\n")





# ─── Step 6: Train & Evaluate ────────────────────────────────
models = {
   "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.5),
   "Decision Tree": DecisionTreeClassifier(random_state=30, max_depth=10, min_samples_split=10, min_samples_leaf=4),
   "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15, min_samples_split=10),
   "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan'),
   "SVM": SVC(random_state=42, kernel='rbf', C=0.8, gamma='scale'),
}

trained_models = {name: model for name, model in models.items()}

print("📊 Training Models...\n")
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    trained_models[name] = model

    print(f"{'='*50}")
    print(f"  {name}  |  Accuracy: {acc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))







# ─── User Input & Prediction ─────────────────────────────────
print("\n🔮 Prediction Mode")
print("Available models:")
for i, name in enumerate(trained_models.keys()):
    print(f"  {i+1}. {name}")

model_choice  = int(input("\nChoose a model (enter number): ")) - 1
chosen_name   = list(trained_models.keys())[model_choice]
chosen_model  = trained_models[chosen_name]
print(f"\n✅ Using: {chosen_name}")

print("\nEnter the following details:")
age            = float(input("Age: "))
education = input("education (e.g. Doctorate, Prof-school, Masters, Bachelors): ")
capital_gain   = float(input("Capital gain: "))
capital_loss   = float(input("Capital loss: "))
hours_per_week = float(input("Hours per week: "))

education_num=education_map.get(education)

print("\nMarital status options:")
marital_options = ["Married-civ-spouse", "Never-married", "Divorced",
                   "Separated", "Widowed", "Married-spouse-absent",
                   "Married-AF-spouse"]

for i, opt in enumerate(marital_options, 1):
    print(f"  {i}. {opt}")
ms_choice      = int(input("Choose marital status (enter number): ")) - 1
marital_status = marital_options[ms_choice]


print("\nRelationship options:")
if marital_status in ["Married-civ-spouse", "Married-AF-spouse"]:
    rel_options = ["Husband", "Wife"]
elif marital_status == "Never-married":
    rel_options = ["Own-child", "Unmarried", "Other-relative"]
else:
    rel_options = ["Unmarried", "Own-child", "Other-relative", "Not-in-family"]

for i, opt in enumerate(rel_options, 1):
    print(f"  {i}. {opt}")
rel_choice   = int(input("Choose relationship (enter number): ")) - 1
relationship = rel_options[rel_choice]

position = input("Occupation (e.g. Tech-support, Craft-repair, Sales): ").strip()





# ─── Build raw DataFrame ─────────────────────────────────────
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



# ─── Apply Same Preprocessing ────────────────────────────────
user_df[cols_to_scale] = scaler.transform(user_df[cols_to_scale])

for col in onehot_cols:
    dummies  = pd.get_dummies(user_df[col], prefix=col).astype(int)
    user_df  = user_df.drop(columns=[col])
    user_df  = pd.concat([user_df, dummies], axis=1)

user_encoded = user_df.reindex(columns=X_train.columns, fill_value=0)



# ─── Predict ─────────────────────────────────────────────────
prediction = chosen_model.predict(user_encoded)
label      = le.inverse_transform(prediction)[0]

print(f"\n💰 Predicted Salary: {label}")

MODEL_DIR = r"C:\Users\ahmed\PycharmProjects\CodeFather\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Save all trained models
for name, model in trained_models.items():
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name.replace(' ', '_')}.pkl"))

# Save scaler and label encoder
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("✅ Models saved successfully!")