from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MOEL_DIR = PROJECT_ROOT /"models"
REPORT_DIR = PROJECT_ROOT /"reports"

features_columns = ["current_power",
    "previous_hour_power",
    "previous_day_power",
    "hour",
    "day_of_week"]

target_column = 'target_next_hour'


def calculate_rmse(actual,predicted):
    return mean_absolute_error(actual,predicted),(((actual-predicted)**2).mean()**0.5)

def main():
    train_data = pd.read_csv(TRAIN_PATH,parse_dates=["datetime"])
    test_data = pd.read_csv(TEST_PATH,parse_dates=["datetime"])

    if train_data["datetime"].max() >= test_data["datetime"].min():
        raise ValueError("Training and testing periods overlap.")

    X_train = train_data[features_columns]
    y_train = train_data[target_column]

    X_test = test_data[features_columns]
    y_test = test_data[target_column]

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Input features:", features_columns)

    baseline_predictions = X_test["current_power"]

    baseline_mae,baseline_rmse = calculate_rmse(y_test,baseline_predictions)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=1,
    )

    model.fit(X_train,y_train)

    prediction = model.predict(X_test)

    model_mae = mean_absolute_error(y_test,prediction)
    model_rmse = ((y_test.to_numpy()-prediction)**2).mean()**0.5
    model_r2 = r2_score(y_test,prediction)

    print("\n===== BASELINE RESULTS =====")
    print(f"MAE:  {baseline_mae:.4f} kW")
    print(f"RMSE: {baseline_rmse:.4f} kW")

    print("\n===== RANDOM FOREST RESULTS =====")
    print(f"MAE:  {model_mae:.4f} kW")
    print(f"RMSE: {model_rmse:.4f} kW")
    print(f"R²:   {model_r2:.4f}")

    print("\n===== COMPARISON =====")

    if model_mae < baseline_mae:
        improvement = ((baseline_mae-model_mae)/baseline_mae)*100
        print(f"MAE improved by {improvement:.2f}% over the baseline.")
    else:
        print("The Random Forest did not improve on baseline MAE.")

    importance = pd.Series(model.feature_importances_,index=features_columns,).sort_values(ascending=False)
    print("\n===== FEATURE IMPORTANCE =====")
    print(importance)

    MOEL_DIR.mkdir(parents=True,exist_ok=True)
    REPORT_DIR.mkdir(parents=True,exist_ok=True)

    model_path = MOEL_DIR/"random_forest_energy_model.joblib"

    joblib.dump(model,model_path)
    results = pd.DataFrame({
        "datetime": test_data["datetime"],
        "actual_next_hour_power": y_test,
        "baseline_prediction": baseline_predictions,
        "random_forest_prediction": prediction,
    })

    results.to_csv(REPORT_DIR/"test_predictions.csv",index=False)

    print("\nModel saved to:", model_path)
    print("Test predictions saved to:", REPORT_DIR / "test_predictions.csv")

if __name__ == "__main__":
    main()