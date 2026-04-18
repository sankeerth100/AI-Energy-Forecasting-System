import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("Starting EDA...")

# Load dataset (IMPORTANT PATH)
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "energy_consumption.csv")

df = pd.read_csv(file_path)

print("\nFirst 5 rows:")
print(df.head())

# Convert datetime column
df["datetime"] = pd.to_datetime(df["datetime"])

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Plot energy trend
plt.figure(figsize=(12,5))
plt.plot(df["datetime"], df["energy"], color="blue")

plt.title("Energy Consumption Over Time")
plt.xlabel("Datetime")
plt.ylabel("Energy Usage")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("energy_trend.png")
print("Graph saved as energy_trend.png")