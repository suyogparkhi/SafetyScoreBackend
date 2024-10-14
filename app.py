from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = FastAPI()

def preprocess_crime_data(file_path):
    crime_data = pd.read_csv(file_path)
    crime_data['timestamp'] = pd.to_datetime(crime_data['timestamp'])
    
    crime_data['day_of_week'] = crime_data['timestamp'].dt.dayofweek
    crime_data['hour_of_day'] = crime_data['timestamp'].dt.hour
    crime_data['year'] = crime_data['timestamp'].dt.year
    crime_data['month'] = crime_data['timestamp'].dt.month
    
    return crime_data

def feature_engineering(crime_data):

    X = crime_data[['latitude', 'longitude', 'day_of_week', 'hour_of_day', 'month', 'year']]
    
    crime_data['severity'] = np.random.randint(1, 10, crime_data.shape[0])
    y = 100 - crime_data['severity'] * 5 
    
    return X, y

def train_and_save_model():
    crime_data = preprocess_crime_data('nagpur_crime_data.csv')
    X, y = feature_engineering(crime_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'safety_score_model.pkl')
    return model

if not os.path.exists('safety_score_model.pkl'):
    model = train_and_save_model()
else:
    model = joblib.load('safety_score_model.pkl')

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

@app.post("/predict_safety_score")
def predict_safety(data: LocationData):
    score = calculate_safety_score(data.latitude, data.longitude, data.timestamp)
    return {"safety_score": score}


if __name__ == "__app__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
