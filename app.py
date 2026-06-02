from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# ==========================================
# Phase 9: FastAPI Deployment Architecture
# ==========================================

app = FastAPI(
    title="Customer Churn Predictor Engine",
    description="Live analytical endpoint to calculate churn volatility risk explicitly in real-time.",
    version="1.0"
)

try:
    # Conceptually loads the tuned estimator natively from Phase 6. 
    # Make sure you inherently save your best pipeline using joblib in modeling.py (e.g., joblib.dump(best_model, 'model.pkl'))
    MODEL = joblib.load("best_churn_model.pkl") 
except Exception:
    MODEL = None

class CustomerFeatures(BaseModel):
    """
    Pydantic Input Mapping. Strongly enforces rigid typing mapping explicitly to the Expected Pipeline Structure.
    Prevents garbage-in/garbage-out phenomena.
    """
    tenure: int
    TotalCharges: str  # Kept as string matching the baseline data before our preprocessing fixes it
    MonthlyCharges: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

@app.get("/")
def health_check():
    return {"status": "Active", "message": "API engine successfully standing by."}

@app.post("/predict")
def calculate_churn_risk(customer: CustomerFeatures):
    """
    Ingests rigid JSON payload bounds dynamically and converts to native pandas structure mapped for Scikit-Learn Pipeline execution.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503, 
            detail="Predictive Estimator Offline. Train and generate 'best_churn_model.pkl' before deploying endpoints."
        )
    
    # 1. Structure the raw data into Pandas Matrix exactly how Phase 1 preprocessor expects
    payload_dict = customer.dict()
    df = pd.DataFrame([payload_dict])
    
    try:
        # 2. Complete Phase 1-6 natively (The Sklearn pipeline applies cleaning -> feat engineering -> encoding automatically)
        probability = MODEL.predict_proba(df)[0][1]
        prediction = int((probability >= 0.5)) # Or apply the Phase 7 Optimized Threshold here ideally!
        
        return {
            "churn_probability": round(float(probability), 4),
            "predicted_class": prediction,
            "business_risk": "High Alert" if probability >= 0.50 else "Stable"
        }
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Algorithmic Execution halted: {str(e)}")

if __name__ == "__main__":
    # Allows starting up dynamically via console using: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
