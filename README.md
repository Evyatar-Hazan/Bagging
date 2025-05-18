# Test Script Failure Risk Prediction

This project loads test execution data from a CSV file, trains a RandomForest model to predict the probability of failure for each script, and presents interactive and dynamic visualizations highlighting high-risk test scripts and trends over time.

### Overview

The main idea is to leverage automated test execution results to identify scripts with a high probability of failure. The project helps QA engineers and test managers focus on the most critical and problematic areas in their test suite.

### rerequisites
- Python 3.8 or higher

- Required Python libraries:

    - pandas

    - scikit-learn

    - plotly

    - matplotlib

    - seaborn

Install dependencies using:

```
pip install -r requirements.txt
```

### requirements.txt

```
pandas>=1.3.0
scikit-learn>=1.0.0
plotly>=5.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Usage

1. Make sure the CSV file all_runs.csv is located inside the csv_reports/ folder.

2. Run the script:

```
python fail_prediction_analysis.py
```

### What Does the Script Do?

- Loads the data and filters for rows with status "PASSED" or "FAILED".

- Adds a binary target column (target) and encodes the script name.

- Trains a RandomForest model to predict failure probability using execution time and script encoding.

- Adds a new column with the predicted failure probability.

- Displays an interactive bar chart showing the top 5 most risky scripts (based on their latest run).

- Displays a line chart showing how the failure probability changes over time for those scripts.

### Code Structure

- ```load_and_prepare_data(filepath)```: Loads and preprocesses the data.

- ```train_model(df)```: Trains the RandomForest model.

- ```add_fail_probability(df, model)```: Adds a predicted failure probability column.

- ```plot_top_risky_scripts(df)```: Shows an interactive bar chart for risky scripts.

- ```plot_fail_probability_trend(df, top_scripts)```: Shows failure probability trends over time.

### License

This project is open for personal or internal customization and extension.


[Presentation](https://gamma.app/docs/-1btv7uqolmjlohp)