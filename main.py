import pandas as pd
import os

from src.feature_engineering import create_features
from src.model_training import train_model
from src.visualization import plot_results

# Load data
df = pd.read_csv("data/energy_consumption.csv")

# Feature engineering
df = create_features(df)

print("Data ready for training")

# Train model
model, X_test, y_test, y_pred = train_model(df)

# Visualization
plot_results(y_test, y_pred)

# Save predictions
os.makedirs("outputs", exist_ok=True)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results.to_csv("outputs/predictions.csv", index=False)

print("Predictions saved at: outputs/predictions.csv")