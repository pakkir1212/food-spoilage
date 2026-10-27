# Food Spoilage Prediction Using Machine Learning

A machine learning project that predicts food spoilage values based on preservation conditions and presents the results through an interactive cloud-deployed Streamlit dashboard.

## Project Overview

Food spoilage is influenced by several factors, including preservation methods, vacuum pressure, contamination level, and cooling temperature. This project uses machine learning regression models to explore the relationship between these factors and the observed spoilage values in a food spoilage experimental dataset.

The project includes data analysis, model training, model evaluation, visualizations, and an interactive web dashboard for making predictions.

## Objectives

- Analyze the food spoilage experimental dataset.
- Train and compare multiple machine learning regression models.
- Select a model based on cross-validation performance.
- Visualize model performance and feature importance.
- Deploy an interactive dashboard using Streamlit.
- Demonstrate a cloud-based machine learning workflow.

## Dataset

The dataset is based on the [Minitab Food Spoilage dataset](https://support.minitab.com/en-us/datasets/doe-data-sets/food-spoilage-data/).

- **Number of observations:** 16
- **Number of original columns:** 10
- **Missing values:** None in the processed dataset
- **Problem type:** Supervised machine learning regression

### Input Features

| Feature | Description |
|---|---|
| Preservative | Preservation formula used (Formula1 or Formula2) |
| VacuumPress | Vacuum pressure setting |
| ContaminationLevel | Experimental contamination level |
| CoolTemp | Cooling temperature setting |

**Target variable:** `Spoilage`

The model estimates the numerical spoilage response associated with the selected experimental conditions.

## Machine Learning Models

The following models were evaluated:

- Gradient Boosting Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Decision Tree Regressor
- Mean Baseline Regressor

Leave-One-Out Cross-Validation (LOOCV) was used to evaluate model performance on this small dataset. Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² were used as evaluation metrics.

### Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | 4.1733 | 5.3763 | 0.6770 |
| Random Forest | 4.4450 | 5.7763 | 0.6271 |
| Extra Trees | 6.3125 | 8.1662 | 0.2548 |
| Decision Tree | 6.3125 | 8.1662 | 0.2548 |
| Mean Baseline | 8.3500 | 10.0903 | -0.1378 |

Gradient Boosting was selected because it achieved the lowest MAE among the evaluated non-baseline models.

## Interactive Dashboard

The Streamlit dashboard provides the following sections:

- **Overview:** Project summary, dataset information, and training metrics.
- **Prediction:** Enter experimental conditions and obtain a spoilage prediction.
- **Model Results:** Explore performance metrics, plots, and result tables.
- **Dataset:** View, filter, and download the processed dataset.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit
- Google Colab for model development
- GitHub for version control and source-code hosting
- Streamlit Community Cloud for deployment

## Project Structure

```text
food-spoilage/
├── app/
│   └── app.py
├── data/
│   └── food_spoilage.csv
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
├── results/
│   ├── feature_importance.csv
│   ├── training_metrics.csv
│   ├── training_predictions.csv
│   ├── table_1_dataset_profile.csv
│   ├── table_2_model_configurations.csv
│   ├── table_3_performance_metrics.csv
│   ├── table_4_predictions.csv
│   ├── table_5_errors.csv
│   ├── table_6_runtime_metrics.csv
│   ├── figure_1_spoilage_distribution.png
│   ├── figure_2_actual_vs_predicted.png
│   ├── figure_3_mae_comparison.png
│   ├── figure_4_rmse_comparison.png
│   ├── figure_5_residuals.png
│   └── figure_6_feature_importance.png
├── src/
│   └── train_model.py
├── requirements.txt
└── README.md
```

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/food-spoilage.git
cd food-spoilage
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

From the repository root directory, run:

```bash
streamlit run app/app.py
```

The dashboard should open in your browser.

**Note:** The application loads the trained model from `models/best_model.joblib` and its metadata from `models/model_metadata.json`. These files must be present in the repository.

## Model Training

To retrain the models, run the training script from the repository root:

```bash
python src/train_model.py
```

The script is designed to train and evaluate the models and save the selected model and evaluation outputs. Check the script and installed dependencies if your local environment behaves differently.

## Cloud Deployment

The application is designed for deployment using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create or select an app connected to your `food-spoilage` repository.
4. Set the main file path to `app/app.py`.
5. Deploy the application.

## Limitations

- The dataset contains only 16 observations.
- Model performance estimates may vary with such a small dataset.
- The model predicts an experimental numerical spoilage response; it does not establish whether food is safe to eat.
- The application does not use live sensors, real-time monitoring, or a live database.
- Predictions should be treated as educational and experimental, not as validated shelf-life or food-safety advice.

## Future Improvements

- Evaluate the model using a larger dataset.
- Validate predictions on independent experimental data.
- Explore additional features and regression algorithms.
- Add database integration for storing predictions.
- Investigate sensor-based data collection and monitoring.
- Improve deployment monitoring and model version management.

## Acknowledgements

Dataset source: [Minitab — Food Spoilage Data Set](https://support.minitab.com/en-us/datasets/doe-data-sets/food-spoilage-data/).

## Author

Developed as a machine learning and cloud computing project.
