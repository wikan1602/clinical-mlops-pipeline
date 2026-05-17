import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Initialize FastAPI
app = FastAPI(
    title="Clinical Readmission API",
    description="API for predicting 30-day hospital readmission risk.",
    version="1.0.0"
)

# 2. Define the Schema (Data Validation)
class PatientRecord(BaseModel):
    age: float = Field(..., example=65.5)
    los: float = Field(..., example=4.2)
    prior_admissions: int = Field(..., example=2)
    gender_binary: int = Field(..., example=1)
    has_social_support: int = Field(..., example=1)
    needs_rehab_or_nursing: int = Field(..., example=0)
    followup_scheduled: int = Field(..., example=1)
    polypharmacy: int = Field(..., example=1)

# 3. Load Model from Registry
# We use the service name 'mlflow' because that's the name in your docker-compose
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Change this in src/api/app.py
MODEL_URI = "models:/Clinical_Readmission_Model/latest" 
# OR use the specific version if you know it (e.g., "models:/Clinical_Readmission_Model/2")

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("✅ API loaded model latest version successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

@app.get("/health")
def health_check():
    """Verify the API and Model Registry are connected."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(patient: PatientRecord):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 1. Check the input
        data_dict = patient.model_dump()
        print(f"DEBUG: Input data: {data_dict}")
        
        # 2. Convert to DataFrame
        input_df = pd.DataFrame([data_dict])
        print(f"DEBUG: DataFrame shape: {input_df.shape}")

        # 3. Predict
        prediction = model.predict(input_df)
        print(f"DEBUG: Raw prediction: {prediction}")
        
        # 4. Return
        return {
            "readmission_prediction": int(prediction[0]),
            "risk_level": "High" if prediction[0] >= 0.5 else "Low",
            "model_version": "1.0.0"
        }
    except Exception as e:
        print(f"❌ CRASH LOG: {str(e)}") # This will show up in your terminal
        raise HTTPException(status_code=500, detail=str(e))