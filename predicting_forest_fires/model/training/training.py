import os
import numpy as np
import pandas as pd
import joblib
import mlflow
from skopt.space import Categorical, Integer, Real
from joblib import Parallel, delayed
from predicting_forest_fires.delta.setup import get_spark_session
from predicting_forest_fires.data.custom import (
    train_logistic_regression_classifier,
    train_decision_tree_classifier,
    train_random_forest_classifier,
    train_mlp_classifier,
)
from predicting_forest_fires.config.config import (
    DELTA_TRAINING_PATH,
    TRAINED_LOGISTIC_REGRESSION_CLASSIFIER,
    TRAINED_DECISION_TREE_CLASSIFIER,
    TRAINED_RANDOM_FOREST_CLASSIFIER,
    TRAINED_MLP_CLASSIFIER,
    MLFLOW_TRACKING_URI,
)

mlflow.set_experiment("Forest Fire Prediction: Model Training")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

spark = get_spark_session()
spark_data = spark.read.format("delta").load(DELTA_TRAINING_PATH)
data: pd.DataFrame = spark_data.toPandas()

target = "area"
features = data.columns[data.columns != target]
X_train = data[features]
y_train = data[target]

logistic_regression_model_param_grid = {
    "C": [0.001, 0.01, 0.1, 0.2],
    "penalty": ["l2"],
    "solver": ["saga"],
}

decision_tree_model_param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 15, 20, 30],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [4, 6, 8],
}

random_forest_search_space = {
    "n_estimators": Integer(200, 500),
    "criterion": Categorical(["gini", "entropy"]),
    "max_depth": Integer(1, 50),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 4),
    "max_features": Categorical(["sqrt", "log2"]),
    "bootstrap": Categorical([True, False]),
    "class_weight": Categorical(["balanced", "balanced_subsample"]),
}

mlp_model_param_grid = {
    "hidden_layer_sizes": [(100, 50, 20), (100, 50), (20, 10), (10, 10), (10,)],
    "activation": ["relu", "logistic"],
    "alpha": [0.001, 0.01, 0.1, 1, 5, 10, 100],
    "learning_rate_init": [0.001, 0.01, 0.1],
    "learning_rate": ["constant", "adaptive"],
    "solver": ["adam"],
}


def parallel_model_training():
    tasks = [
        delayed(train_logistic_regression_classifier)(
            X_train,
            y_train,
            logistic_regression_model_param_grid,
        ),
        delayed(train_decision_tree_classifier)(
            X_train,
            y_train,
            decision_tree_model_param_grid,
        ),
        delayed(train_random_forest_classifier)(
            X_train,
            y_train,
            random_forest_search_space,
        ),
        delayed(train_mlp_classifier)(
            X_train,
            y_train,
            mlp_model_param_grid,
        ),
    ]

    trained_models = Parallel(n_jobs=6, verbose=10)(tasks)
    return trained_models


# Training each model in parallel
if __name__ == "__main__":
    trained_models_output = parallel_model_training()


# Save each trained model (locally) to its respective path
joblib.dump(trained_models_output[0], TRAINED_LOGISTIC_REGRESSION_CLASSIFIER)
joblib.dump(trained_models_output[1], TRAINED_DECISION_TREE_CLASSIFIER)
joblib.dump(trained_models_output[2], TRAINED_RANDOM_FOREST_CLASSIFIER)
joblib.dump(trained_models_output[3], TRAINED_MLP_CLASSIFIER)
