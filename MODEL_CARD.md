# Model Card — Customer Churn Classifier

## Model Overview

| Property         | Value                                      |
|------------------|--------------------------------------------|
| Model Type       | Logistic Regression                        |
| Task             | Binary Classification (Churn / No Churn)  |
| Framework        | scikit-learn (sklearn Pipeline)            |
| Serialization    | joblib (.pkl)                              |
| Version          | 1.0                                        |
| Last Updated     | June 2025                                  |

---

## Intended Use

This model is designed to help telecom companies identify customers who are likely to cancel their subscription (churn) in the near future. Predictions can be used to trigger proactive retention campaigns such as discount offers or outreach calls.

**Intended users:** Customer success teams, business analysts, product managers at telecom companies.

**Out-of-scope uses:** This model should not be used for credit scoring, employment decisions, or any high-stakes automated decisions without human review.

---

## Dataset

| Property         | Value                                          |
|------------------|------------------------------------------------|
| Name             | IBM Telco Customer Churn Dataset               |
| Source           | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Size             | 7,043 customers, 21 features                  |
| Target Variable  | `Churn` (Yes / No)                            |
| Class Balance    | ~73% No Churn, ~27% Churn (imbalanced)        |

---

## Features Used

| Feature            | Type        | Description                             |
|--------------------|-------------|-----------------------------------------|
| `tenure`           | Numeric     | Months the customer has been with the company |
| `MonthlyCharges`   | Numeric     | Monthly bill amount in USD              |
| `TotalCharges`     | Numeric     | Cumulative total billed                 |
| `SeniorCitizen`    | Numeric     | 1 if senior citizen, 0 otherwise        |
| `gender`           | Categorical | Male / Female                           |
| `Partner`          | Categorical | Yes / No                                |
| `Dependents`       | Categorical | Yes / No                                |
| `InternetService`  | Categorical | DSL / Fiber optic / No                  |
| `Contract`         | Categorical | Month-to-month / One year / Two year    |
| `PaymentMethod`    | Categorical | Electronic check / Mailed check / etc.  |
| `OnlineSecurity`   | Categorical | Yes / No / No internet service          |

---

## Preprocessing

All preprocessing is handled inside a single `sklearn.Pipeline`:

- **Numeric features:** Standardized using `StandardScaler` (zero mean, unit variance)
- **Categorical features:** One-hot encoded using `OneHotEncoder(handle_unknown="ignore")`

---

## Performance Metrics

Evaluated on training data (no separate test split in v1.0):

| Metric      | No Churn | Churn |
|-------------|----------|-------|
| Precision   | ~0.84    | ~0.65 |
| Recall      | ~0.90    | ~0.52 |
| F1 Score    | ~0.87    | ~0.58 |
| **Accuracy**| **~0.81**|       |

> ⚠️ These metrics are on training data. A proper train/test split evaluation is recommended before production deployment.

---

## Classification Threshold

The default threshold is **35% churn probability** (lower than the standard 50%) to prioritize catching at-risk customers early.

| Probability      | Prediction     | Risk Level | Recommended Action                    |
|------------------|----------------|------------|---------------------------------------|
| ≥ 65%            | Will Churn     | High       | Offer retention discount immediately  |
| 35% – 64%        | Will Churn     | Medium     | Schedule a follow-up call             |
| < 35%            | Will Not Churn | Low        | No action needed                      |

---

## Limitations

- Model was trained on a single historical dataset from one telecom company; may not generalize to others
- Class imbalance (~27% churn) means the model may underperform on minority-class predictions
- No temporal validation — customer behaviour changes over time and the model should be retrained periodically
- Features like `OnlineSecurity` are self-reported and may be inaccurate

---

## Ethical Considerations

- Gender is included as a feature; fairness across gender groups has not been formally evaluated
- The model should be used as a **decision support tool**, not as the sole basis for customer treatment decisions
- Customers should not be penalized or denied services based on model predictions

---

## How to Retrain

```bash
python retrain.py
```

This re-reads the raw data, rebuilds the full pipeline, and overwrites `models/churn_classifier.pkl`.

---

## Author

**Harsha Esmayil**  
[GitHub](https://github.com/harshaesmayil)