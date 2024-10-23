# handler.py

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess_transform.preprocessing import preprocess_data
from model_training.train import train_model
from predict_inference.inference import evaluate_model
from preprocess_transform.transform import transform
import pandas as pd

columns_to_drop = ['Vehicle_ID', 'Year_of_Manufacture', 'Last_Maintenance_Date', 'Maintenance_Type',
                   'Maintenance_Cost', 'Brake_Condition', 'Oil_Quality', 'Failure_History',
                   'Delivery_Times', 'Downtime_Maintenance', 'Impact_on_Efficiency', 'Engine_Temperature','Vehicle_Type','Battery_Status']

# Function to load the trained model from a pickle file
def load_model():
    try:
        with open('trained_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print("Model not found! Please retrain the model first.")
        return None


# Function to retrain the model
def retrain_model():
    print("Starting model training...")
    
    

# Drop irrelevant features before splitting the data
    # Step 1: Load and preprocess data
    data = pd.read_csv('data.csv')  # Load data
    data, _ = preprocess_data(data)  # Preprocess it
    X = data.drop(columns=['Maintenance_Required'] + columns_to_drop)
    y = data['Maintenance_Required']
    
    # Step 2: Transform the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = transform(X_train, fit_scaler=True)
    X_test = transform(X_test)
    
    model = train_model(X_train, y_train)
    
    # Step 4: Save the trained model
    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    print("Model retrained and saved as 'trained_model.pkl'.")
    
    evaluate_model(model, X_test, y_test)




# Function to handle single input prediction
def predict_single_input():
    # Step 1: Load the trained model
    model = load_model()
    if model is None:
        return
    
    # Step 2: Take user input for prediction
    
    #for this instance i have taken a random row from the dataset itself
    
    data = pd.read_csv('data.csv')
    
    user_input_df = data.sample(1)
    
    # Step 3: Preprocess and transform the input
    
    user_input, _ = preprocess_data(user_input_df)
    user_input = user_input.drop(columns=['Maintenance_Required'] + columns_to_drop)
    user_input_transformed = transform(user_input, fit_scaler=True)
    
    # Step 4: Make a prediction
    prediction = model.predict(user_input_transformed)
    
    print(f"Prediction result for the given input: {prediction[0]}")
    

# Function to show menu and handle user prompts
def main():
    while True:
        print("\nSelect an action:")
        print("1. Retrain the model")
        print("2. Predict using single input")
        print("3. Exit")
        
        choice = input("Enter the number of your choice: ").strip()
        
        if choice == '1':
            retrain_model()
        elif choice == '2':
            predict_single_input()
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()
