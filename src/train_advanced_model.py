from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT /"data"/"processed"/"household_power_advanced_features.csv")

MODEL_DIR = PROJECT_ROOT/"models"
REPORT_DIR = PROJECT_ROOT/"reports"

features_columns  =[ "current_power",
    "previous_hour_power",
    "previous_day_power",
    "rolling_mean_3h",
    "rolling_std_6h",
    "hour",
    "day_of_week",]
target_columns = "target_next_hour"

def main():
    data = pd.read_csv(DATA_PATH,parse_dates=["datetime"])
    data  = data.sort_values("datetime").reset_index(drop=True)

    split = int(len(data)*0.80)
    train_data = data.iloc[:split]
    test_data = data.iloc[split:]

    X_train = train_data[features_columns]
    y_train = train_data[target_columns]
    X_test = test_data[features_columns]
    y_test = test_data[target_columns]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
        min_samples_leaf=2,
    )

    model.fit(X_train,y_train)
    prediction = model.predict(X_test)

    baseline_prediction = X_test["current_power"]

    baseline_mae = mean_absolute_error(y_test,baseline_prediction)
    baseline_rmse = mean_squared_error(y_test,baseline_prediction) ** 0.5

    model_mae = mean_absolute_error(y_test,prediction)
    model_rmse = mean_squared_error(y_test,prediction)**0.5
    model_r2 = r2_score(y_test,prediction)

    print("===== ADVANCED MODEL RESULTS =====")
    print("Training samples:", len(train_data))
    print("Testing samples:", len(test_data))
    print(f"Baseline MAE: {baseline_mae:.4f} kW")
    print(f"Baseline RMSE: {baseline_rmse:.4f} kW")
    print(f"Advanced MAE: {model_mae:.4f} kW")
    print(f"Advanced RMSE: {model_rmse:.4f} kW")
    print(f"Advanced R²: {model_r2:.4f}")

    improvement = (baseline_mae-model_mae) / baseline_mae *100
    print(f"MAE improvement over baseline: {improvement:.2f}%")

    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    REPORT_DIR.mkdir(parents=True,exist_ok=True)

    model_path = MODEL_DIR /"advanced_random_forest.joblib"
    joblib.dump(model,model_path)

    output = test_data[["datetime"]].copy()
    output["actual_next_hour_power"] = y_test.to_numpy()
    output["baseline_prediction"] = baseline_prediction.to_numpy()
    output["advanced_prediction"] = prediction
    output.to_csv(REPORT_DIR / "advanced_test_predictions.csv")

if __name__ == "__main__":
    main()