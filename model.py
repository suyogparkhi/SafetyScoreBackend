import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

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

def train_model(file_path):
    crime_data = preprocess_crime_data(file_path)
    X, y = feature_engineering(crime_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Model MSE: {mse}')
    
    joblib.dump(model, 'safety_score_model.pkl')

if __name__ == "__main__":
    train_model('nagpur_crime_data.csv')
