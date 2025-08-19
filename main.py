from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model at startup
model = joblib.load("model.pkl")

app = FastAPI(title="ML Model API", description="API for Iris classification")

# Input schema
class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float = None

# Label mapping
iris_classes = ["setosa", "versicolor", "virginica"]

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        features = np.array([[input_data.sepal_length,
                              input_data.sepal_width,
                              input_data.petal_length,
                              input_data.petal_width]])
        
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features).max()

        return PredictionOutput(
            prediction=iris_classes[prediction],
            confidence=float(proba)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "Classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": iris_classes
    }
