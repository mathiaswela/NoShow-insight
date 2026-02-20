import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_and_save_model(data_path: str, model_save_path: str):
    print(f"Loading cleaned data from {data_path}...")
    df = pd.read_csv(data_path)

    # Defining X and y
    print("Splitting data into train and test sets...")
    X = df.drop(columns=['NoShow_numeric'])
    y = df['NoShow_numeric']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)} rows. Test set: {len(X_test)} rows.")

    #Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=10,
                                   class_weight='balanced',
                                   random_state=42
                                   )
    model.fit(X_train, y_train)

    # Model eval
    print("\n -- Model Evaluation -- \n")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model for production
    print(f"Saving model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Dumping model
    joblib.dump(model, model_save_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    print("Starting model training pipeline...")

    INPUT_DATA = 'data/noshow_cleaned_production.csv'
    MODEL_SAVE_PATH = 'models/noshow_rf_model.joblib'

    train_and_save_model(INPUT_DATA, MODEL_SAVE_PATH)
    print("Model training pipeline completed successfully!")
