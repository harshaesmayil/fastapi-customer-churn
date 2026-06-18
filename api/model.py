import joblib
import pandas as pd

# Load model
model = joblib.load(r"C:\Users\harsh\OneDrive\Desktop\fastapi_customer_churn\models\churn_classifier.pkl")

# Load raw data for scaling
df_raw = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\fastapi_customer_churn\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
df_raw["TotalCharges"] = df_raw["TotalCharges"].fillna(0)
df_raw["SeniorCitizen"] = pd.to_numeric(df_raw["SeniorCitizen"], errors="coerce")

# Load processed data for column order
df_ref = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\fastapi_customer_churn\data\processed\processed_churn.csv")
feature_columns = df_ref.drop("Churn", axis=1).columns

def scale(value, col):
    return (value - df_raw[col].mean()) / df_raw[col].std(ddof=0)

def build_sample(c):
    internet = c.internet_service.lower()
    contract = c.contract.lower()
    payment  = c.payment_method.lower()
    security = c.online_security.lower()

    sample = {
        'SeniorCitizen'                          : scale(c.senior_citizen, 'SeniorCitizen'),
        'tenure'                                 : scale(c.tenure, 'tenure'),
        'MonthlyCharges'                         : scale(c.monthly_charges, 'MonthlyCharges'),
        'TotalCharges'                           : scale(c.total_charges, 'TotalCharges'),

        'gender_Male'                            : 1 if c.gender == "Male" else 0,
        'Partner_Yes'                            : 1 if c.partner == "Yes" else 0,
        'Dependents_Yes'                         : 1 if c.dependents == "Yes" else 0,

        'PhoneService_Yes'                       : 1,
        'MultipleLines_No phone service'         : 0,
        'MultipleLines_Yes'                      : 0,

        'InternetService_Fiber optic'            : 1 if internet == "fiber optic" else 0,
        'InternetService_No'                     : 1 if internet == "no" else 0,

        'OnlineSecurity_No internet service'     : 1 if internet == "no" else 0,
        'OnlineSecurity_Yes'                     : 1 if security == "yes" else 0,

        'OnlineBackup_No internet service'       : 1 if internet == "no" else 0,
        'OnlineBackup_Yes'                       : 0,

        'DeviceProtection_No internet service'   : 1 if internet == "no" else 0,
        'DeviceProtection_Yes'                   : 0,

        'TechSupport_No internet service'        : 1 if internet == "no" else 0,
        'TechSupport_Yes'                        : 0,

        'StreamingTV_No internet service'        : 1 if internet == "no" else 0,
        'StreamingTV_Yes'                        : 0,

        'StreamingMovies_No internet service'    : 1 if internet == "no" else 0,
        'StreamingMovies_Yes'                    : 0,

        'Contract_One year'                      : 1 if contract == "one year" else 0,
        'Contract_Two year'                      : 1 if contract == "two year" else 0,

        'PaperlessBilling_Yes'                   : 1,

        'PaymentMethod_Credit card (automatic)'  : 1 if payment == "credit card (automatic)" else 0,
        'PaymentMethod_Electronic check'         : 1 if payment == "electronic check" else 0,
        'PaymentMethod_Mailed check'             : 1 if payment == "mailed check" else 0,
    }

    df = pd.DataFrame([sample])
    return df[feature_columns]

def get_risk_and_action(prob):
    if prob >= 0.60:
        risk = "Critical High Risk"
        action = "Offer retention discount immediately"
    elif prob >= 0.35:
        risk = "High Risk"
        action = "Schedule customer satisfaction call"
    elif prob >= 0.20:
        risk = "Medium Risk"
        action = "Monitor account activity closely"
    else:
        risk = "Low Risk"
        action = "No action needed, customer is stable"
    return risk, action

def predict_single(customer):
    sample = build_sample(customer)
    prob = model.predict_proba(sample)[0][1]
    prediction = "Will Churn" if prob >= 0.35 else "Will Stay"
    risk, action = get_risk_and_action(prob)
    return {
        "prediction": prediction,
        "probability": round(float(prob), 4),
        "risk_level": risk,
        "recommended_action": action
    }

def predict_batch(customers):
    return [predict_single(c) for c in customers]