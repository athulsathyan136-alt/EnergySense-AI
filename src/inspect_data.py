from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt")

def load_data(path):
    data = pd.read_csv(
        path,sep=";",
        na_values=["?"],
        low_memory=False,
    )
    return data

def inspect_data(data):
    print('\n======DATASET OVERVIEW=========')
    print('Rows and Columns: ',data.shape)
    print('\n======COLUMN NAMES=========')
    print(data.columns.tolist())
    print('\n======FIRST FIVE ROWS=========')
    print(data.head())
    print('\n======DATA TYPES=========')
    print(data.dtypes)
    print('\n======MISSING VALUES=========')
    print(data.isnull().sum())

def main():
    if not DATA_PATH.exists():
        print('Dataset not found!')
        print('Expected location: ',DATA_PATH)
        return

    data = load_data(DATA_PATH)
    inspect_data(data)

if __name__ == "__main__":
    main()