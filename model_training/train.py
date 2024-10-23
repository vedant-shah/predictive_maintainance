from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    # Handle class imbalance using SMOTE
    smote = SMOTE(sampling_strategy=0.7, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_res, y_train_res)
    
    return model
