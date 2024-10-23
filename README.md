# Predictive Maintenance Model

This project implements a predictive maintenance solution using a machine learning pipeline. The goal is to build a model capable of predicting when maintenance is required for vehicles based on several factors such as vehicle load, engine temperature, and anomalies detected by IoT sensors.

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Inputs and Outputs](#inputs-and-outputs)
- [Execution Steps](#execution-steps)
- [Model Details](#model-details)

## Project Overview

This project splits the machine learning workflow into three main stages:

1. **Preprocessing and Transformation:** Loading the data, cleaning, feature engineering, and splitting.
2. **Model Training:** Training a machine learning model using the preprocessed data.
3. **Inference and Results:** Making predictions, evaluating model performance, and generating the output.

The code is organized to make it easy to maintain, test, and deploy. After training, the final model is saved as a `.pkl` (pickle) file for future predictions.

## Folder Structure

```
|-- data.csv                     # Input data file
|-- handler.py                   # Main script to execute the entire pipeline
|-- model_training/
|   |-- train.py                 # Training of the Random Forest classifier
|-- pm_1.ipynb                   # Jupyter notebook for exploratory data analysis
|-- predict_inference/
|   |-- inference.py             # Inference and evaluation of the model
|-- preprocess_transform/
|   |-- preprocessing.py         # Data preprocessing logic
|   |-- transform.py             # Data transformation logic
|-- README.md                    # Documentation and usage guide
|-- requirements.txt             # Python package dependencies
|-- trained_model.pkl            # Pickle file for the trained model (generated after training)
```

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/vedant-shah/predictive_maintainance.git
cd predictive-maintenance
```

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

This will install all the required packages including:

- `pandas`: For data manipulation.
- `numpy`: For numerical computations.
- `scikit-learn`: For machine learning models and data preprocessing.
- `imbalanced-learn`: For handling class imbalance using SMOTE.

## Usage

### Step 1: Prepare the Input Data

- Place the CSV file containing vehicle maintenance data (`data.csv`) in the project directory.
- Ensure the CSV has the following columns:
  - **Categorical Features**: `Make_and_Model`, `Vehicle_Type`, `Route_Info`, `Weather_Conditions`, `Road_Conditions`
  - **Numerical Features**: `Usage_Hours`, `Load_Capacity`, `Actual_Load`, `Engine_Temperature`, `Tire_Pressure`, `Fuel_Consumption`, `Battery_Status`, `Vibration_Levels`, `Anomalies_Detected`, `Predictive_Score`, `Downtime_Maintenance`
  - **Datetime Features**: `Last_Maintenance_Date`
  - **Target Variable**: `Maintenance_Required` (whether the vehicle needs maintenance or not)

### Step 2: Execute the Pipeline

To execute the full machine learning pipeline (data preprocessing, model training, evaluation, and generating the model pickle file), run the following command:

```bash
python handler.py
```

This script will:

- Preprocess and transform the data using scripts in the `preprocess_transform` directory.
- Train the model using the script in the `model_training` directory.
- Evaluate the model and make predictions using the script in the `predict_inference` directory.
- Save the trained model as `trained_model.pkl`.

## Inputs and Outputs

### Inputs

- **Input Data**: `data.csv` containing vehicle maintenance data.

### Outputs

- **Trained Model**: `trained_model.pkl` file containing the trained machine learning model.
- **Evaluation Metrics**: Printed to the console during the evaluation phase.

## Execution Steps

1. **Preprocessing and Transformation**: Run the preprocessing and transformation scripts to clean and prepare the data.
2. **Model Training**: Train the machine learning model using the preprocessed data.
3. **Inference and Results**: Evaluate the model performance and make predictions.

## Model Details

The model used in this project is a Random Forest classifier. The training script is located in the `model_training/train.py` file, and the evaluation script is located in the `predict_inference/inference.py` file.
