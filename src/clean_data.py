from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt")

OUTPUT_PATH = (PROJECT_ROOT /"data"/"processed"/'household_power_clean.csv')

def load_data(path):
    data = pd.read_csv(path,sep=";",na_values=["?"],low_memory=False)
    return data
def clean_data(data):
    data["datetime"] = pd.to_datetime(data["Date"]+" "+data["Time"],
                                      format="%d/%m/%Y %H:%M:%S",errors='coerce',)

    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column],errors='coerce')

    data = data.sort_values('datetime')
    data = data.dropna(subset=["datetime","Global_active_power"])
    data = data.drop(columns=["Date","Time"])

    data = data.reset_index(drop=True)

    return data
def main():
    if not DATA_PATH.exists():
        print('Dataset not found: ',DATA_PATH)
        return
    data = load_data(DATA_PATH)
    print("Original dataset shape ",data.shape)
    print('Missing active-power values: ',data["Global_active_power"].isna().sum(),)

    cleaned_data = clean_data(data)
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    cleaned_data.to_csv(OUTPUT_PATH,index=False)

    print("\n=== CLEANING RESULTS ===")
    print("Cleaned dataset shape:", cleaned_data.shape)
    print("Missing values after cleaning:")
    print(cleaned_data.isna().sum())
    print("\nFirst five cleaned rows:")
    print(cleaned_data.head())
    print("\nSaved cleaned data to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()