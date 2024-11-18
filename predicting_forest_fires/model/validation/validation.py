import os
import numpy as np
import pandas as pd
import mlflow
from predicting_forest_fires.delta.setup import get_spark_session
from predicting_forest_fires.data.custom import (
    load_trained_model,
    model_validation,
)
from predicting_forest_fires.config.config import (
    DELTA_TRAINING_PATH,
    MODEL_VALIDATION_REPORT,
    MLFLOW_TRACKING_URI,
)

mlflow.set_experiment("Forest Fire Prediction: Model Validation")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

(
    logistic_regression_model,
    decision_tree_model,
    random_forest_model,
    mlp_model,
) = load_trained_model()

spark = get_spark_session()
spark_data = spark.read.format("delta").load(DELTA_TRAINING_PATH)
training_data: pd.DataFrame = spark_data.toPandas()

target = "area"
features = training_data.columns[training_data.columns != target]
X_train = training_data[features]
y_train = training_data[target]

with mlflow.start_run(run_name="Model Validation Operation Run"):
    logistic_regressor_model_validation = model_validation(
        logistic_regression_model,
        X_train,
        y_train,
        "Logistic Regression Classifier",
    )

    decision_tree_model_validation = model_validation(
        decision_tree_model,
        X_train,
        y_train,
        "Decision Tree Classifier",
    )

    random_forest_model_validation = model_validation(
        random_forest_model,
        X_train,
        y_train,
        "Random Forest Classifier",
    )

    mlp_model_validation = model_validation(
        mlp_model,
        X_train,
        y_train,
        "MLP Classifier",
    )

    combined_validation = pd.concat(
        [
            logistic_regressor_model_validation,
            decision_tree_model_validation,
            random_forest_model_validation,
            mlp_model_validation,
        ],
        axis=0,
        ignore_index=True,
    )

    combined_validation_path = MODEL_VALIDATION_REPORT
    combined_validation.to_csv(combined_validation_path, index=False)
    mlflow.log_artifact(
        combined_validation_path,
        artifact_path="validation_reports",
    )
    os.remove(combined_validation_path)
