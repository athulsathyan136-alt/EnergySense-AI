from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "test_predictions.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

def main():
    data = pd.read_csv(PREDICTIONS_PATH,parse_dates=["datetime"],)

    actual = data["actual_next_hour_power"]
    predicted = data["random_forest_prediction"]

    mae = mean_absolute_error(actual,predicted)
    rmse = mean_squared_error(actual,predicted)**0.5

    print("===== MODEL EVALUATION =====")
    print(f"Test records: {len(data)}")
    print(f"MAE: {mae:.4f} kW")
    print(f"RMSE: {rmse:.4f} kW")

    sample = data.head(168)

    plt.figure(figsize=(14,6))
    plt.plot(sample["datetime"],
             sample["actual_next_hour_power"],
             label="Actual consumption")

    plt.plot(sample["datetime"],
             sample["random_forest_prediction"],
             label="Random Forest Prediction",)


    plt.title("EnergySense-AI: Actual vs Predicted Electricity Consumption")
    plt.xlabel("Date and time")
    plt.ylabel("Active power (kW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    chart_path = REPORT_DIR/"actual_vs_predicted.png"
    plt.savefig(chart_path,dpi=150)
    plt.close()
    print("\nGraph saved to:", chart_path)

if __name__ == "__main__":
    main()
