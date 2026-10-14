
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "household_power_hourly.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "household_power_advanced_features.csv"
)


def main():
    data = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
    data = data.sort_values("datetime").set_index("datetime")

    # Restore every hourly timestamp, including missing hours.
    full_index = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq="h",
    )
    data = data.reindex(full_index)
    data.index.name = "datetime"

    power = data["Global_active_power"]

    features = pd.DataFrame(index=data.index)

    features["current_power"] = power
    features["previous_hour_power"] = power.shift(1)
    features["previous_day_power"] = power.shift(24)

    # Calculate statistics using only earlier readings.
    features["rolling_mean_3h"] = (
        power.shift(1).rolling(window=3).mean()
    )
    features["rolling_std_6h"] = (
        power.shift(1).rolling(window=6).std()
    )

    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek

    # Predict exactly one clock hour into the future.
    features["target_next_hour"] = power.shift(-1)

    features = features.dropna()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index_label="datetime")

    print("===== ADVANCED FEATURE DATASET =====")
    print("Rows:", len(features))
    print("Columns:", len(features.columns))
    print("Missing values:", features.isna().sum().sum())
    print("Saved to:", OUTPUT_PATH)
    print("\nColumns:")
    print(features.columns.tolist())


if __name__ == "__main__":
    main()
