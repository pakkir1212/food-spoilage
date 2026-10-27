
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data" / "food_spoilage.csv"
MODELS_DIR = PROJECT / "models"
RESULTS_DIR = PROJECT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)

    target = "Spoilage"
    features = [
        "Preservative",
        "VacuumPress",
        "ContaminationLevel",
        "CoolTemp",
    ]

    required = features + [target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].dropna().reset_index(drop=True)
    X = df[features]
    y = df[target].astype(float)

    categorical = ["Preservative"]
    numerical = [
        "VacuumPress",
        "ContaminationLevel",
        "CoolTemp",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
            (
                "numerical",
                SimpleImputer(strategy="median"),
                numerical,
            ),
        ]
    )

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=3, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=3, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100, max_depth=3, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            random_state=42
        ),
    }

    cv = LeaveOneOut()
    metric_rows = []
    prediction_rows = []
    fitted_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocess)),
            ("model", clone(estimator)),
        ])

        start = time.perf_counter()
        predictions = cross_val_predict(
            pipeline, X, y, cv=cv, n_jobs=None
        )
        elapsed = time.perf_counter() - start

        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        metric_rows.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "LOOCV_Runtime_Seconds": elapsed,
        })

        for i, (actual, predicted) in enumerate(zip(y, predictions)):
            prediction_rows.append({
                "Observation": i + 1,
                "Model": name,
                "Actual": float(actual),
                "Predicted": float(predicted),
                "Absolute_Error": abs(float(actual - predicted)),
            })

        # Fit a separate deployment model on all available observations.
        # This model is for future demonstrations, not independent validation.
        if name != "Mean Baseline":
            full_pipeline = clone(pipeline)
            full_pipeline.fit(X, y)
            fitted_models[name] = full_pipeline

    metrics = pd.DataFrame(metric_rows).sort_values("MAE")
    metrics.to_csv(RESULTS_DIR / "training_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        RESULTS_DIR / "training_predictions.csv", index=False
    )

    # Select the lowest-MAE non-baseline model.
    eligible = metrics[metrics["Model"] != "Mean Baseline"]
    best_name = eligible.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    metadata = {
        "selected_model": best_name,
        "target": target,
        "features": features,
        "evaluation": "Leave-One-Out Cross-Validation",
        "observations": int(len(df)),
        "warning": (
            "Preliminary results from a very small dataset; "
            "not validated for food-safety decisions."
        ),
    }

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nLOOCV model comparison:")
    print(metrics.to_string(index=False))
    print(f"\nSelected model by LOOCV MAE: {best_name}")
    print(f"Saved model: {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved metadata: {MODELS_DIR / 'model_metadata.json'}")
    print("\nNote: performance estimates are preliminary (small dataset).")


if __name__ == "__main__":
    main()
