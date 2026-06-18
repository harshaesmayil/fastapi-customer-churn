import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ── 2. Clean ──────────────────────────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ── 3. Define features ────────────────────────────────────────────────────────
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents",
    "InternetService", "Contract", "PaymentMethod", "OnlineSecurity"
]

TARGET = "Churn"

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

# ── 4. Build the pipeline ─────────────────────────────────────────────────────
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# ── 5. Train ──────────────────────────────────────────────────────────────────
pipeline.fit(X, y)

# ── 6. Evaluate (optional but useful) ────────────────────────────────────────
from sklearn.metrics import accuracy_score, classification_report
y_pred = pipeline.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))

# ── 7. Save the pipeline (replaces old model) ─────────────────────────────────
joblib.dump(pipeline, "models/churn_classifier.pkl")
print("Pipeline saved to models/churn_classifier.pkl")