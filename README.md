# EnergySense-AI ⚡

**Household Electricity Consumption Forecasting Using Machine Learning**

EnergySense-AI is an end-to-end machine learning project that predicts household electricity consumption one hour ahead using historical power readings and time-based features.

The project uses the UCI Individual Household Electric Power Consumption dataset and compares a simple baseline prediction with Random Forest regression models.

## 🎯 Project Objectives

- Process and clean real-world electricity consumption data.
- Explore consumption patterns and missing timestamps.
- Engineer historical and rolling statistical features.
- Train models using a chronological train/test split.
- Evaluate prediction accuracy with MAE, RMSE, and R².
- Investigate prediction errors and visualize model performance.

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains household electricity measurements collected over multiple years.

The project focuses on `Global_active_power`, measured in kilowatts (kW).

The original dataset is not included in this repository. Download it from the source above and place `household_power_consumption.txt` inside `data/raw/`.

## 🧠 Machine Learning Approach

### Baseline

The baseline predicts that the next hour's power consumption will equal the current hour's consumption.

### Random Forest Models

The models use historical consumption and time-based features to predict next-hour power consumption.

The advanced feature set includes:

- Current power consumption
- Previous-hour power consumption
- Previous-day power consumption
- Three-hour rolling mean
- Six-hour rolling standard deviation
- Hour of day
- Day of week

Rolling features are calculated from previous observations to avoid using future readings as input.

### Evaluation

Models are evaluated using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of determination

The data is split chronologically so later observations are held out for testing.

## 📈 Current Advanced Model Results

Results from the current run:

| Metric | Baseline | Advanced Random Forest |
|---|---:|---:|
| MAE | 0.3982 kW | 0.3470 kW |
| RMSE | 0.6045 kW | 0.4983 kW |
| R² | — | 0.5555 |

The advanced model reduced MAE by approximately **12.87%** compared with its baseline on its test set.

**Comparison note:** The original and advanced feature datasets contain different numbers of rows, so their test periods are not perfectly matched. A comparison on identical test timestamps would provide a stricter assessment.

These results are from a local experiment and have not been independently reproduced.

## 🛠️ Technology Stack

- Python
- pandas
- NumPy
- Matplotlib
- scikit-learn
- joblib
- Git and GitHub

## 📁 Project Structure

```text
EnergySense-AI/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── eda.py
│   ├── hourly_data.py
│   ├── check_hourly_gaps.py
│   ├── create_features.py
│   ├── create_advanced_features.py
│   ├── split_data.py
│   ├── train_model.py
│   ├── train_advanced_model.py
│   ├── evaluate_model.py
│   ├── analyze_errors.py
│   └── investigate_worst_error.py
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Setup

Clone the repository:

```bash
git clone https://github.com/athulsathyan136-alt/EnergySense-AI.git
cd EnergySense-AI
```

Create and activate a virtual environment on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the UCI dataset and place the raw text file in `data/raw/`.

## ▶️ Run the Pipeline

Run these scripts from the project root, in order:

```powershell
python .\src\inspect_data.py
python .\src\clean_data.py
python .\src\eda.py
python .\src\hourly_data.py
python .\src\check_hourly_gaps.py
python .\src\create_features.py
python .\src\split_data.py
python .\src\train_model.py
python .\src\evaluate_model.py
python .\src\analyze_errors.py
python .\src\investigate_worst_error.py
python .\src\create_advanced_features.py
python .\src\train_advanced_model.py
```

Run the scripts in this order because later steps depend on files produced by earlier steps.

## 🔮 Future Improvements

- Compare both models on exactly the same test timestamps.
- Tune hyperparameters using time-aware validation.
- Add automated tests and a reproducible training pipeline.
- Build a FastAPI prediction service.
- Containerize the service with Docker.
- Explore cloud deployment and model