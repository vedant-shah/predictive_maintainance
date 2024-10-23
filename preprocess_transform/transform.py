# transform.py
from sklearn.preprocessing import StandardScaler

# Define the scaler globally so that it's consistent across all steps
scaler = StandardScaler()

# Function to transform data (applicable for training, test, or single input)
def transform(data, fit_scaler=False):
    print("Transforming data...")
    if fit_scaler:
        # Fit and transform the data for training
        x = scaler.fit_transform(data)

        print("Data transformation completed.")
        return x
    else:
        # Only transform the data for test or single input
        x = scaler.transform(data)

        print("Data transformation completed.")
        return x
