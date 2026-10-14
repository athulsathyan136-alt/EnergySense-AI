
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "test_predictions.csv"
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    data = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["datetime"],
    )

    # Calculate the absolute size of each prediction mistake.
    data["error"] = (
        data["random_forest_prediction"]
        - data["actual_next_hour_power"]
    )
    data["absolute_error"] = data["error"].abs()

    # Find the row with the largest mistake.
    worst_index = data["absolute_error"].idxmax()
    worst = data.loc[worst_index]
    worst_time = worst["datetime"]

    print("===== WORST PREDICTION =====")
    print("Timestamp:", worst_time)
    print(f"Actual consumption: {worst['actual_next_hour_power']:.4f} kW")
    print(f"Predicted consumption: {worst['random_forest_prediction']:.4f} kW")
    print(f"Signed error: {worst['error']:.4f} kW")

    # Select available prediction records within 12 hours either side.
    start_time = worst_time - pd.Timedelta(hours=12)
    end_time = worst_time + pd.Timedelta(hours=12)

    window = data[
        (data["datetime"] >= start_time)
        & (data["datetime"] <= end_time)
    ].copy()

    print("\n===== NEARBY PREDICTIONS =====")
    print(
        window[
            [
                "datetime",
                "actual_next_hour_power",
                "random_forest_prediction",
                "error",
            ]
        ].to_string(index=False)
    )

    # Plot actual and predicted values around the event.
    plt.figure(figsize=(13, 6))
    plt.plot(
        window["datetime"],
        window["actual_next_hour_power"],
        marker=".",
        label="Actual consumption",
    )
    plt.plot(
        window["datetime"],
        window["random_forest_prediction"],
        marker=".",
        label="Random Forest prediction",
    )
    plt.axvline(
        worst_time,
        linestyle="--",
        label="Worst-error timestamp",
    )

    plt.title("EnergySense-AI: Investigation of Largest Prediction Error")
    plt.xlabel("Date and time")
    plt.ylabel("Active power (kW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_path = REPORT_DIR / "worst_prediction_investigation.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print("\nInvestigation graph saved to:", chart_path)


if __name__ == "__main__":
    main()
