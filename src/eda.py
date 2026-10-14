from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (PROJECT_ROOT /"data"/"processed"/"household_power_clean.csv")

REPORTS_DIR = PROJECT_ROOT / "reports"


def load_data(path):
    data = pd.read_csv(path,parse_dates=["datetime"])
    return data
def explore_data(data):
    power = data["Global_active_power"]

    print("\n=== DATASET INFORMATION ===")
    print("Rows and colums: ",power.shape )
    print("First timestamp : ",data["datetime"].min())
    print("Last timestamp: ",data["datetime"].max() )

    print("\n=== ACTIVE POWER STATISTICS (kW) ===")
    print('Average: ',round(power.mean(),3))
    print('Median: ',round(power.median(),3))
    print('Minimun : ',round(power.min(),3))
    print('Maximum: ',round(power.max(),3))
    print('Standard deviation : ',round(power.std(),3))

    print("\n=== HOURLY AVERAGE CONSUMPTION ===")
    hours = (data.set_index("datetime")["Global_active_power"].resample("h").mean())
    print(hours.head(10))

    return hours

def create_plots(data,hours):
    REPORTS_DIR.mkdir(parents=True,exist_ok=True)

    sample = data.iloc[:1440]

    plt.figure(figsize=(12,5))

    plt.plot(sample["datetime"],
             sample["Global_active_power"],)

    plt.title('Household Electicity Consumption: First 24 hours')
    plt.xlabel("Time")
    plt.ylabel("Active Power(kW)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "first_24_hours.png")
    plt.close()
    first_week = hours.iloc[:168]

    plt.figure(figsize=(12, 5))
    plt.plot(first_week.index, first_week.values)
    plt.title("Hourly Average Electricity Consumption: First Seven Days")
    plt.xlabel("Time")
    plt.ylabel("Average Active Power (kW)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "first_week_hourly.png")
    plt.close()

    print("\nGraphs saved in:", REPORTS_DIR)

def main():
    if not DATA_PATH.exists():
        print("Cleaned dataset not found: ",DATA_PATH)
        print("Run python src/clean_data.py first")
        return

    data = load_data(DATA_PATH)

    hours = explore_data(data)
    create_plots(data,hours)

if __name__ == "__main__":
    main()