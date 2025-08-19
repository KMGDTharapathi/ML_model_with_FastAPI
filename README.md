# ML Model API (Iris Classification)

## Problem
Predict iris flower species (Setosa, Versicolor, Virginica) based on 4 numerical features.

## Model
- Algorithm: RandomForestClassifier
- Dataset: sklearn Iris dataset

## API Endpoints
- `GET /` → Health check
- `POST /predict` → Get prediction
- `GET /model-info` → Model details

## Example Request (POST /predict)
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
"# ML_model_with_FastAPI" 
