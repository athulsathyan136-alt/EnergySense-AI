from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (PROJECT_ROOT /"data" /"processed"/"household_power_features.csv")

TRAIN_PATH= PROJECT_ROOT /"data"/"processed"/"train.csv"
TEST_PATH = PROJECT_ROOT /"data"/"processed"/"test.csv"

def main():
    data = pd.read_csv(INPUT_PATH,parse_dates=["datetime"])
    data = data.sort_values("datetime").reset_index(drop=True)

    if not data["datetime"].is_monotonic_increasing:
        raise ValueError("The timestamps are not sorted correctly.")

    if len(data) < 10:
        raise ValueError("Not enough rows to split the dataset.")
    
    split_posistion = int(len(data)*0.80)

    train_data = data.iloc[:split_posistion].copy()
    test_data = data.iloc[split_posistion:].copy()

    train_data.to_csv(TRAIN_PATH,index=False)
    test_data.to_csv(TEST_PATH,index=False)

    print("Total rows:", len(data))
    print("Training rows:", len(train_data))
    print("Testing rows:", len(test_data))

    print("\nTraining period:")
    print(train_data["datetime"].min(), "to", train_data["datetime"].max())

    print("\nTesting period:")
    print(test_data["datetime"].min(), "to", test_data["datetime"].max())
    if train_data["datetime"].max() >= test_data["datetime"].min():
        raise ValueError("Training and testing periods overlap.")
    
    print("\nSaved training data to:", TRAIN_PATH)
    print("Saved testing data to:", TEST_PATH)

if __name__ == "__main__":
    main()