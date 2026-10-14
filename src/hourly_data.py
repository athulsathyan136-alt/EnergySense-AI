from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (PROJECT_ROOT / "data" / "processed" / "household_power_clean.csv")

OUTPUT_PATH = (PROJECT_ROOT / "data" / "processed" / "household_power_hourly.csv")

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT_PATH}\n")


    data = pd.read_csv(INPUT_PATH,parse_dates=["datetime"],)

    print("Cleaned data shape: ",data.shape)
    print('\nFirst five rows: ')
    print(data.head())

    data = data.sort_values("datetime")
    data = data.set_index("datetime")

    hourly_data = (data["Global_active_power"].resample("h").mean().to_frame())

    missing_hours = hourly_data["Global_active_power"].isna().sum()

    hourly_data = hourly_data.dropna()

    hourly_data.to_csv(OUTPUT_PATH)

    print("\nHourly data shape:", hourly_data.shape)
    print("Hours without valid readings:", missing_hours)

    print("\nFirst five hourly averages:")
    print(hourly_data.head())

    print("\nLast five hourly averages:")
    print(hourly_data.tail())

    print("\nSaved hourly data to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
