from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np

model = joblib.load('safety_score_model.pkl')

app = FastAPI()

class LocationData(BaseModel):
    latitude: float
    longitude: float
    timestamp: str 

def calculate_safety_score(lat, lon, time):
    time = datetime.fromisoformat(time)
    
    day_of_week = time.weekday()
    hour_of_day = time.hour
    year = time.year
    month = time.month

    feature_vector = np.array([[lat, lon, day_of_week, hour_of_day, month, year]])

    score = model.predict(feature_vector)
    return max(0, min(100, score[0]))  


@app.get("/")
def read_root():
    return {"message": "Safety Score API is up and running!"}

# Endpoint to predict safety score
@app.post("/predict_safety_score")
def predict_safety(data: LocationData):
    score = calculate_safety_score(data.latitude, data.longitude, data.timestamp)
    return {"safety_score": score}
