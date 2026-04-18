import pandas as pd
import numpy as np

print("Dataset generation started...")

# Create hourly timestamps (realistic energy data)
date_range = pd.date_range(start="2023-01-01", periods=2000, freq="h")

# Set randomness seed for same output every time
np.random.seed(42)

# Simulate energy consumption pattern
energy = (
    200
    + 50 * np.sin(np.arange(2000) * 0.01)   # seasonal pattern
    + np.random.normal(0, 20, 2000)         # noise
)

# Create dataframe
df = pd.DataFrame({
    "datetime": date_range,
    "energy": energy
})

# Save file
df.to_csv("energy_consumption.csv", index=False)

print("Dataset created successfully!")
print(df.head())