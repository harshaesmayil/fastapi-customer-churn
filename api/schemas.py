from pydantic import BaseModel
from typing import Literal

class CustomerInput(BaseModel):
    senior_citizen: int                    # 0 or 1
    tenure: int                            # months
    monthly_charges: float
    total_charges: float
    gender: Literal["Male", "Female"]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    online_security: Literal["Yes", "No"]

class CustomerOutput(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    recommended_action: str

class BatchInput(BaseModel):
    customers: list[CustomerInput]