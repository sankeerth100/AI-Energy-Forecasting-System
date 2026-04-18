import matplotlib
matplotlib.use("Agg")   # 👈 THIS FIXES YOUR ERROR

import matplotlib.pyplot as plt
import os

def plot_results(y_test, y_pred):

    # ensure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(10,5))

    plt.plot(y_test.values, label="Actual Energy", color="blue")
    plt.plot(y_pred, label="Predicted Energy", color="red")

    plt.title("Actual vs Predicted Energy Consumption")
    plt.xlabel("Samples")
    plt.ylabel("Energy Usage")

    plt.legend()
    plt.tight_layout()

    plt.savefig("outputs/actual_vs_predicted.png")
    print("Graph saved: outputs/actual_vs_predicted.png")