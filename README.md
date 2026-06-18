#  Customer Churn Prediction API

A machine learning REST API built with **FastAPI** that predicts whether a telecom customer will churn based on their account and service details.

---

##  Project Overview

Customer churn is one of the biggest challenges in the telecom industry. This project trains a **Logistic Regression** model on real telecom customer data and serves predictions through a production-style REST API with input validation, batch prediction support, and a live Swagger UI.

---

##  Screenshots

### Swagger UI — `/predict`
<!-- Add screenshot of single prediction request and response here -->
<img width="1422" height="832" alt="image" src="https://github.com/user-attachments/assets/4ce51fdd-4ca3-4582-9b49-f6de1b303753" />

### Swagger UI — `/predict/batch`
<!-- Add screenshot of batch prediction request and response here -->
request:
<img width="1420" height="600" alt="image" src="https://github.com/user-attachments/assets/d704dc22-f490-44d4-96dc-903dac6b53dc" />
<img width="1432" height="572" alt="image" src="https://github.com/user-attachments/assets/f45d2ef3-5b96-40f9-8bf3-6e7cc10eb98e" />
<img width="1422" height="591" alt="image" src="https://github.com/user-attachments/assets/d8ed8c97-d342-4495-a4fe-5d5f07791747" />

response:
<img width="1421" height="856" alt="image" src="https://github.com/user-attachments/assets/39911ae2-ef54-4d8d-bf45-3515f3fb4417" />

---

##  Project Structure

```
fastapi_customer_churn/
│
├── api/
│   ├── main.py          # FastAPI app, routes
│   ├── model.py         # Model loading, preprocessing, prediction logic
│   └── schemas.py       # Pydantic input/output validation schemas
│
├── data/
│   ├── raw/             # Original Telco dataset
│   └── processed/       # Cleaned and processed CSV
│
├── models/
│   └── churn_classifier.pkl   # Trained Logistic Regression model (joblib)
│
├── retrain.py           # Script to retrain the model from scratch
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Model Details

| Property | Value |
|---|---|
| Algorithm | Logistic Regression |
| Dataset | IBM Telco Customer Churn |
| Features | Tenure, Monthly Charges, Contract Type, Internet Service, Payment Method, and more |
| Churn Threshold | 35% probability |
| Serialization | joblib |

### Preprocessing
- Numerical features scaled using training set means
- Categorical features one-hot encoded (contract type, internet service, payment method, gender, etc.)

### Classification Logic
```
Churn Probability ≥ 35%  →  "Will Churn"
Churn Probability < 35%  →  "Will Not Churn"
```

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/harshaesmayil/fastapi-customer-churn.git
cd fastapi-customer-churn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API
```bash
cd api
uvicorn main:app --reload
```

### 4. Open Swagger UI
```
http://localhost:8000/docs
```

---

##  API Endpoints

### `GET /`
Health check — confirms the API is running.

### `GET /health`
Checks if the model is loaded and ready to serve predictions.

### `POST /predict`
Predict churn for a single customer.

**Request Body:**
```json
{
  "senior_citizen": 0,
  "tenure": 12,
  "monthly_charges": 65.5,
  "total_charges": 540.0,
  "gender": "Male",
  "partner": "Yes",
  "dependents": "No",
  "internet_service": "Fiber optic",
  "contract": "Month-to-month",
  "payment_method": "Electronic check",
  "online_security": "Yes"
}
```

**Response:**
```json
{
  "churn_prediction": "Will Churn",
  "churn_probability": 0.6394,
  "risk_level": "High",
  "recommended_action": "Offer retention discount immediately"
}
```

### `POST /predict/batch`
Predict churn for multiple customers in one request.

**Request Body:**
```json
{
  "customers": [
    { "tenure": 3, "monthly_charges": 85.0, "contract": "Month-to-month", ... },
    { "tenure": 60, "monthly_charges": 20.0, "contract": "Two year", ... }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    { "churn_prediction": "Will Churn", "churn_probability": 0.71, ... },
    { "churn_prediction": "Will Not Churn", "churn_probability": 0.12, ... }
  ]
}
```

---

##  Testing

You can test the API using:

**Swagger UI** — `http://localhost:8000/docs` (recommended)

**curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "senior_citizen": 0,
    "tenure": 3,
    "monthly_charges": 85.0,
    "total_charges": 255.0,
    "gender": "Male",
    "partner": "No",
    "dependents": "No",
    "internet_service": "Fiber optic",
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "online_security": "No"
  }'
```

**Python:**
```python
import requests

data = {
  "senior_citizen": 0,
  "tenure": 3,
  "monthly_charges": 85.0,
  "total_charges": 255.0,
  "gender": "Male",
  "partner": "No",
  "dependents": "No",
  "internet_service": "Fiber optic",
  "contract": "Month-to-month",
  "payment_method": "Electronic check",
  "online_security": "No"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

---

##  Retraining the Model

To retrain the model on fresh or updated data:

```bash
python retrain.py
```

This will reprocess the data, retrain the Logistic Regression model, and overwrite `models/churn_classifier.pkl`.

---

##  Requirements

```
fastapi
uvicorn
scikit-learn
pandas
numpy
joblib
pydantic
```

Install all with:
```bash
pip install -r requirements.txt
```

---

##  Dataset

**IBM Telco Customer Churn Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customers, 21 features
- Target variable: `Churn` (Yes/No)

---

##  Author

**Harsha Esmayil**  
[GitHub](https://github.com/harshaesmayil)
