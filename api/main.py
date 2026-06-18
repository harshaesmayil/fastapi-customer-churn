from fastapi import FastAPI
from api.schemas import CustomerInput, CustomerOutput, BatchInput
from api.model import predict_single, predict_batch
from datetime import datetime

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn or not",
    version="1.0.0"
)

# Store predictions in memory
history = []

@app.get("/")
def root():
    return {
        "message": "Customer Churn API is running",
        "model": "Logistic Regression",
        "accuracy": "82.1%"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=CustomerOutput)
def predict(customer: CustomerInput):
    result = predict_single(customer.dict())

    # Save to history
    history.append({
        "id": len(history) + 1,
        "input": customer.dict(),
        "result": result,
        "timestamp": datetime.now().isoformat()
    })

    return result

@app.post("/predict/batch")
def batch_predict(data: BatchInput):
    results = predict_batch([c.dict() for c in data.customers])

    # Save each to history
    for customer, result in zip(data.customers, results):
        history.append({
            "id": len(history) + 1,
            "input": customer.dict(),
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    return {
        "total": len(results),
        "predictions": results
    }

@app.get("/history")
def get_history():
    return {
        "total_predictions": len(history),
        "predictions": history
    }

@app.get("/stats")
def get_stats():
    if not history:
        return {"message": "No predictions made yet"}

    total = len(history)
    will_churn = sum(1 for h in history if h["result"]["prediction"] == "Will Churn")
    will_stay = total - will_churn
    critical = sum(1 for h in history if h["result"]["risk_level"] == "Critical High Risk")
    high = sum(1 for h in history if h["result"]["risk_level"] == "High Risk")

    return {
        "total_predictions": total,
        "will_churn": will_churn,
        "will_stay": will_stay,
        "churn_rate": f"{round((will_churn / total) * 100, 1)}%",
        "critical_high_risk": critical,
        "high_risk": high
    }