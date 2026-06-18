import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])

y = df["Churn"]
X = df.drop("Churn", axis=1)
numerical_cols = X.select_dtypes(exclude="object").columns
X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

processed_df = pd.concat([X, y], axis=1)
processed_df.to_csv("data/processed/processed_churn.csv", index=False)
print("Saved processed_churn.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")

joblib.dump(model, "models/churn_classifier.pkl")
print("Model saved!")