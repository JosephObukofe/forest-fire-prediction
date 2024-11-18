import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
from predicting_forest_fires.delta.setup import get_spark_session
from predicting_forest_fires.data.custom import (
    load_trained_model,
    model_prediction,
    model_confusion_matrix,
    model_classification_report,
)
from predicting_forest_fires.config.config import (
    DELTA_TEST_PATH,
    CONFUSION_MATRIX,
    MODEL_EVALUATION_REPORT,
    MLFLOW_TRACKING_URI,
)

mlflow.set_experiment("Forest Fire Prediction: Model Evaluation")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

(
    logistic_regression_model,
    decision_tree_model,
    random_forest_model,
    mlp_model,
) = load_trained_model()

spark = get_spark_session()
spark_data = spark.read.format("delta").load(DELTA_TEST_PATH)
test_data: pd.DataFrame = spark_data.toPandas()

target = "area"
features = test_data.columns[test_data.columns != target]
X_test = test_data[features]
y_test = test_data[target]

model_predictions = {
    "Logistic Regression Classifier": model_prediction(
        logistic_regression_model, X_test
    ),
    "Decision Tree Classifier": model_prediction(decision_tree_model, X_test),
    "Random Forest Classifier": model_prediction(random_forest_model, X_test),
    "MLP Classifier": model_prediction(mlp_model, X_test),
}

with mlflow.start_run(run_name="Model Evaluation Operation Run"):
    for model_name, y_pred in model_predictions.items():
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=1)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=1)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=1)
        mlflow.log_metrics(
            {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (model_name, y_pred) in enumerate(model_predictions.items()):
        model = [
            logistic_regression_model,
            decision_tree_model,
            random_forest_model,
            mlp_model,
        ][i]
        model_confusion_matrix(model, y_test, y_pred, ax=axes[i])
        axes[i].set_title(model_name)

    plt.tight_layout()
    confusion_matrix_path = CONFUSION_MATRIX
    plt.savefig(confusion_matrix_path)
    mlflow.log_artifact(
        confusion_matrix_path,
        artifact_path="confusion_matrices",
    )
    os.remove(confusion_matrix_path)

    model_evaluation_report = model_classification_report(model_predictions, y_test)
    model_evaluation_report_path = MODEL_EVALUATION_REPORT
    model_evaluation_report.to_csv(model_evaluation_report_path, index=False)
    mlflow.log_artifact(
        model_evaluation_report_path,
        artifact_path="evaluation_reports",
    )
    os.remove(model_evaluation_report_path)
