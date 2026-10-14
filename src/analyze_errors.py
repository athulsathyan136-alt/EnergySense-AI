from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "test_predictions.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

def main():
    data = pd.read_csv(PREDICTIONS_PATH,parse_dates=["datetime"],)

    data["error"] = (data["random_forest_prediction"] - data["actual_next_hour_power"])

    data["absolute_error"] = data["error"].abs()
    print("===== ERROR ANALYSIS =====")
    print(f"Average signed error: {data['error'].mean():.4f} kW")
    print(f"Average absolute error: {data['absolute_error'].mean():.4f} kW")
    print(f"Largest absolute error: {data['absolute_error'].max():.4f} kW")

    print("\n===== 5 LARGEST PREDICTION ERRORS =====")
    print(data.nlargest(5,"absolute_error")[["datetime",
                "actual_next_hour_power",
                "random_forest_prediction",
                "error",
                "absolute_error",]].to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.hist(data["error"], bins=50, edgecolor="black")
    plt.axvline(0, linestyle="--", label="Zero error")
    plt.title("EnergySense-AI: Prediction Error Distribution")
    plt.xlabel("Prediction error (kW)")
    plt.ylabel("Number of predictions")
    plt.legend()
    plt.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = REPORT_DIR / "prediction_error_distribution.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print("\nError chart saved to:", chart_path)


if __name__ == "__main__":
    main()
