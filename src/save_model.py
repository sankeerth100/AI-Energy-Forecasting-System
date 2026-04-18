import joblib
import os

def save_model(model):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/energy_model.pkl")

    print("Model saved at: models/energy_model.pkl")