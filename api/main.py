from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from src.utils import predict_nb, predict_nn

app = FastAPI(title="Employee Attrition API")


# Input schema
class EmployeeData(BaseModel):
    features: List[float]


@app.get("/")
def home():
    return {"message": "Attrition Prediction API Running"}


# Naive Bayes prediction
from fastapi import HTTPException

@app.post("/predict/nb")
def predict_naive_bayes(data: EmployeeData):
    try:
        prediction = predict_nb(data.features)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Neural Network prediction
@app.post("/predict/nn")
def predict_neural_network(data: EmployeeData):
    prediction = predict_nn(data.features, input_size=len(data.features))
    return {"model": "Neural Network", "prediction": prediction}