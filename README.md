# Food Spoilage Prediction Using Machine Learning

## Overview

A research prototype that compares machine-learning regression models
for predicting the numerical `Spoilage` target from a small experimental
food-spoilage dataset.

## Features

- Streamlit dashboard with overview, prediction, model-results, and dataset tabs.
- Five-model comparison: Mean Baseline, Decision Tree, Random Forest,
  Extra Trees, and Gradient Boosting.
- Leave-One-Out Cross-Validation (LOOCV) evaluation.
- Saved model pipeline and model metadata.
- Experiment graphs, evaluation tables, and prediction outputs.

## Dataset

The project uses the Minitab Food Spoilage DOE dataset:
https://support.minitab.com/en-us/datasets/doe-data-sets/food-spoilage-data/

The dataset contains 16 observations. Inputs are Preservative, VacuumPress,
ContaminationLevel, and CoolTemp. The numerical target is Spoilage.

## Run locally

Install dependencies:

    pip install -r requirements.txt

Train and evaluate the models:

    python src/train_model.py

Launch the dashboard:

    streamlit run app/app.py

## Current results

The latest recorded LOOCV comparison selected Gradient Boosting by MAE:

- MAE: approximately 4.173
- RMSE: approximately 5.376
- R2: approximately 0.677

These results are preliminary and based on only 16 observations.
They are not independent validation and do not establish food-safety reliability.

## Limitations

This prototype does not currently provide validated food-safety decisions,
live sensor monitoring, temporal forecasting, or operational alerts.
Advanced model modules remain future work until implemented and evaluated.
