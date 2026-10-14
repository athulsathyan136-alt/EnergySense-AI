## Exploratory Data Analysis

The project uses the UCI Individual Household Electric Power Consumption dataset.

### Initial findings
- Loaded more than two million original electricity readings.
- Removed rows with missing active-power measurements.
- Converted separate date and time columns into a datetime column.
- Aggregated minute-level active-power readings into hourly averages.
- Generated graphs to inspect consumption patterns over time.

### Visualizations
- `reports/first_24_hours.png` — first 24 hours of minute-level electricity consumption.
- `reports/first_week_hourly.png` — hourly average consumption during the first seven days.

### Tools
- Python
- pandas
- Matplotlib

**Note:** These graphs cover the beginning of the dataset. They are exploratory visualizations, not evidence that the same patterns occur throughout the entire period.