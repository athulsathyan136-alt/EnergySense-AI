from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (PROJECT_ROOT /"data" / "processed" / "household_power_hourly.csv")

def main():
    data = pd.read_csv(INPUT_PATH,parse_dates=["datetime"],index_col="datetime",)

    data = data.sort_index()

    excepted_hours = pd.date_range(start=data.index.min(),end=data.index.max(),freq="h",)

    missing_hours = excepted_hours.difference(data.index)

    print("First timestamp:", data.index.min())
    print("Last timestamp:", data.index.max())
    print("Valid hourly records:", len(data))
    print("Expected hourly timestamps:", len(excepted_hours))
    print("Missing timestamps:", len(missing_hours))

    if len(excepted_hours) > 0:
        missing_percentage = (len(missing_hours) / len(excepted_hours)) * 100
        print(f"Missing percentage: {missing_percentage:2f}%")

    print("\nFirst 10 missing timestamps:")
    print(missing_hours[:10])

    if len(missing_hours) > 0:
        gaps = pd.Series(missing_hours)
        gap_groups = (gaps.diff().ne(pd.Timedelta(hours=1))).cumsum()

        longest_gap = (gaps.groupby(gap_groups).size().max())

        print("\nLongest consecutive gap:", longest_gap, "hours")
    else:
        print("\nNo missing hourly timestamps found.")

if __name__ == "__main__":
    main()
