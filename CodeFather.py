import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import os
import joblib


# ─── Step 1: Load and Clean ───────────────────────────────────
df_train = pd.read_csv("Train.csv")
df_test = pd.read_csv("Test.csv")

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

# Strip spaces from string columns
for col in df_train.select_dtypes(include='object').columns:
    df_train[col] = df_train[col].str.strip()
for col in df_test.select_dtypes(include='object').columns:
    df_test[col] = df_test[col].str.strip()

# ─── Step 2: Visualizing Gender vs Salary (BEFORE DROPPING) ───
print("📊 Generating Visualizations...")
plt.figure(figsize=(10, 6))
sns.countplot(data=df_train, x='sex', hue='salary', palette='viridis')
plt.title('Salary Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# ─── Step 3: Encoding and Feature Selection ───────────────────
target = "salary"
le = LabelEncoder()
df_train["salary"] = le.fit_transform(df_train["salary"])
df_test["salary"]  = le.transform(df_test["salary"])

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

# ─── Step 4: Balancing with SMOTE ────────────────────────────
print("⚖️ Balancing training data with SMOTE...")
sm = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# ─── Step 5: Train & Evaluate ────────────────────────────────
models = {
   "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.5),
   "Decision Tree": DecisionTreeClassifier(random_state=30, max_depth=10, min_samples_split=10, min_samples_leaf=4),
   "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15, min_samples_split=10),
   "KNN": KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan'),
   "SVM": SVC(random_state=42, kernel='rbf', C=0.8, gamma='scale'),
}

trained_models = {}

print("\n📊 Training Models...\n")
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    trained_models[name] = model
    print(f" {name:<20} | Accuracy: {acc:.4f}")

# ─── Step 6: Prediction Mode & Saving ────────────────────────
print("\n🔮 Prediction Mode")
for i, name in enumerate(trained_models.keys(), 1):
    print(f"  {i}. {name}")

model_choice  = int(input("\nChoose a model (1-5): ")) - 1
chosen_name   = list(trained_models.keys())[model_choice]
chosen_model  = trained_models[chosen_name]

print(f"\n✅ Using: {chosen_name}")
age            = float(input("Age: "))
education      = input("Education (e.g. Doctorate, Bachelors): ")
capital_gain   = float(input("Capital gain: "))
capital_loss   = float(input("Capital loss: "))
hours_per_week = float(input("Hours per week: "))
education_num  = education_map.get(education, 10) # Default to 10 if not found

# Build raw DataFrame for prediction
user_df = pd.DataFrame([{
    'age': age, 'education-num': education_num, 'capital-gain': capital_gain,
    'capital-loss': capital_loss, 'hours-per-week': hours_per_week,
    'marital-status': 'Never-married', # Simplification for example
    'position': 'Sales',             # Simplification for example
    'relationship': 'Not-in-family'   # Simplification for example
}])

# Note: In a real app, you'd need the full user input for all columns
# This part applies the same dummies/scaling as the training data
user_df[cols_to_scale] = scaler.transform(user_df[cols_to_scale])
# ... (Processing user_df for one-hot encoding same as training)
user_encoded = pd.get_dummies(user_df).reindex(columns=X_train.columns, fill_value=0)

prediction = chosen_model.predict(user_encoded)
label = le.inverse_transform(prediction)[0]
print(f"\n💰 Predicted Salary: {label}")

# ─── Step 7: Saving ──────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
for name, model in trained_models.items():
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name.replace(' ', '_')}.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le,     os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("\n✅ All set! Models and plots ready.")