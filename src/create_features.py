
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
    / "household_power_features.csv"
)


def main():
    # Step 1: Load the hourly dataset.
    hourly_data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["datetime"],
        index_col="datetime",
    )

    hourly_data = hourly_data.sort_index()

    # Step 2: Build a complete hourly timeline.
    full_index = pd.date_range(
        start=hourly_data.index.min(),
        end=hourly_data.index.max(),
        freq="h",
    )

    # Missing hours become NaN; we do not invent readings.
    hourly_data = hourly_data.reindex(full_index)
    hourly_data.index.name = "datetime"

    power = hourly_data["Global_active_power"]

    # Step 3: Create the features and next-hour target.
    features = pd.DataFrame(index=hourly_data.index)

    features["current_power"] = power
    features["previous_hour_power"] = power.shift(1)
    features["previous_day_power"] = power.shift(24)

    # Calendar information known at prediction time.
    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek

    # The value we want to predict is one hour in the future.
    features["target_next_hour"] = power.shift(-1)

    # Step 4: Keep only rows with all required measurements.
    required_columns = [
        "current_power",
        "previous_hour_power",
        "previous_day_power",
        "target_next_hour",
    ]

    features = features.dropna(subset=required_columns)

    # Step 5: Save the prepared dataset.
    features.to_csv(OUTPUT_PATH)

    # Step 6: Display results.
    print("Feature dataset shape:", features.shape)

    print("\nFirst five feature rows:")
    print(features.head())

    print("\nLast five feature rows:")
    print(features.tail())

    print("\nMissing values:")
    print(features.isna().sum())

    print("\nSaved features to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
