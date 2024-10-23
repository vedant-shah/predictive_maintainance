# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Function for general preprocessing (applicable to training, test, or single input data)
def preprocess_data(data):
    
    print("Preprocessing data...")
        
    categorical_features = ['Make_and_Model', 'Vehicle_Type', 'Route_Info', 'Weather_Conditions', 'Road_Conditions']
    
    numerical_features = ['Usage_Hours', 'Load_Capacity', 'Actual_Load', 'Engine_Temperature', 'Tire_Pressure','Fuel_Consumption', 'Battery_Status', 'Vibration_Levels', 'Anomalies_Detected','Predictive_Score', 'Downtime_Maintenance']
    
    
    

    # Encode categorical features using Label Encoding
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        data[feature] = label_encoders[feature].fit_transform(data[feature])
        
    # Fill missing categorical values with the mode
    for feature in categorical_features:
        data[feature] = data[feature].fillna(data[feature].mode()[0])

    # Fill missing numerical values with the mean
    for feature in numerical_features:
        data[feature] = data[feature].fillna(data[feature].mean())

    # Convert 'Last_Maintenance_Date' to datetime format
    data['Last_Maintenance_Date'] = pd.to_datetime(data['Last_Maintenance_Date'], errors='coerce')

    # Derived Features
    data['Time_Since_Last_Maintenance'] = (pd.to_datetime('today') - data['Last_Maintenance_Date']).dt.days
    data['Vehicle_Age'] = pd.to_datetime('today').year - data['Year_of_Manufacture']
    data['Load_Utilization'] = (data['Actual_Load'] / data['Load_Capacity']) * 100

    
    print("Data preprocessing completed.")
    return data, label_encoders
