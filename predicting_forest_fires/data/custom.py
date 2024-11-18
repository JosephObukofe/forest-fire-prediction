import os
import time
import numpy as np
import pandas as pd
import joblib
import minio
import urllib.parse
import shutil
import logging
import subprocess
import mlflow
import mlflow.sklearn
import mlflow.tracking
from mlflow.pyfunc import PyFuncModel
from mlflow.exceptions import MlflowException
from datetime import datetime
from minio import Minio
from minio.error import MinioException, S3Error
from typing import List, Any, Dict, Tuple, Optional, Union
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from skopt import BayesSearchCV
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from predicting_forest_fires.config.config import (
    TRAINED_LOGISTIC_REGRESSION_CLASSIFIER,
    TRAINED_DECISION_TREE_CLASSIFIER,
    TRAINED_RANDOM_FOREST_CLASSIFIER,
    TRAINED_MLP_CLASSIFIER,
    MODEL_VALIDATION_DATA,
    EVALUATION_TEST_DATA,
    TRAINING_PREPROCESSOR,
    OPTIMIZED_FEATURES,
    MINIO_URL,
    MINIO_BUCKET,
    MINIO_PATH,
    MINIO_OBJECT_NAME,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
    MLFLOW_TRACKING_URI,
)


logger = logging.getLogger(__name__)


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies either a standard log transformation or a modified log transformation
    to handle both positively and negatively skewed data.

    Parameters
    ----------
    transform_type : str, default="log_transform"
        Specifies the type of log transformation to apply. Options are:
        - "log_transform": Applies a standard log transformation log(x + 1).
        - "modified_log_transform": Applies a modified log transformation to handle negative values
          by shifting the data with a constant (abs(min) + 1).

    Attributes
    ----------
    shifting_constant_ : float
        The constant used to shift the data for the modified log transform to handle negative values.
    """

    def __init__(self, transform_type="log_transform"):
        self.transform_type = transform_type
        self.shifting_constant_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if self.transform_type == "modified_log_transform":
            self.shifting_constant_ = abs(X.min()) + 1
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if self.transform_type == "log_transform":
            X_transformed = np.log(X + 1)
        elif self.transform_type == "modified_log_transform":
            if self.shifting_constant_ is None:
                raise ValueError("The transformer has not been fitted yet.")
            X_shifted = X + self.shifting_constant_
            X_transformed = np.log(X_shifted)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"feature_{i}" for i in range(len(input_features))])
        suffix = (
            "_log_transformed"
            if self.transform_type == "log_transform"
            else "_modified_log_transformed"
        )
        return np.array([f"{feature}{suffix}" for feature in input_features])


def preprocess_training_set(
    X: pd.DataFrame,
    log_transform_features: List[str],
    modified_log_transform_features: List[str],
    numerical_features: List[str],
    categorical_features: List[str],
) -> Tuple[pd.DataFrame, Pipeline]:
    """
    Preprocesses the input DataFrame by applying log transformation to specified features,
    one-hot encoding for categorical features, and then scales numerical features using StandardScaler.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing features to be preprocessed.
    log_transform_features : List[str]
        List of features that need standard log transformation.
    modified_log_transform_features : List[str]
        List of features that need modified log transformation.
    numerical_features : List[str]
        List of numerical features that need scaling.
    categorical_features : List[str]
        List of categorical features that need one-hot encoding.

    Returns
    -------
    Tuple[pd.DataFrame, Pipeline]
        A tuple where:
        - pd.DataFrame: A new DataFrame with transformed features.
        - Pipeline: A fitted pipeline that includes the preprocessing steps.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "log_transform",
                LogTransformer(transform_type="log_transform"),
                log_transform_features,
            ),
            (
                "modified_log_transform",
                LogTransformer(transform_type="modified_log_transform"),
                modified_log_transform_features,
            ),
            (
                "scaling",
                RobustScaler(),
                numerical_features,
            ),
            (
                "encoding",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(steps=[("preprocessing", preprocessor)])
    processed_array = pipeline.fit_transform(X)
    transformed_column_names = pipeline.named_steps[
        "preprocessing"
    ].get_feature_names_out()
    processed_df = pd.DataFrame(processed_array, columns=transformed_column_names)
    return processed_df, pipeline


def preprocess_test_set(
    X: pd.DataFrame,
    preprocessor: BaseEstimator,
) -> pd.DataFrame:
    """
    Preprocesses the test set using a previously fitted pipeline and transformer.

    This function applies transformations learned during training to the test set.
    It does not fit or impute any information from the test set, thus preventing data leakage.

    Parameters
    ----------
    X : pd.DataFrame
        The input test DataFrame containing features to be preprocessed.
    preprocessor : BaseEstimator
        A previously fitted scikit-learn transformer or pipeline used for preprocessing.
        This preprocessor has been trained on the training data and will be used to transform the test set.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the preprocessed test set, with the same transformed features as in the training set.
    """

    processed_array = preprocessor.transform(X)
    transformed_column_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(processed_array, columns=transformed_column_names)
    return processed_df


def preprocess_inference_set(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the inference data using the same preprocessor and feature selection
    applied during the training phase.

    This function performs several steps:
    1. Geographical encoding of spatial features.
    2. Binning of continuous features into binary values.
    3. Transformation using the pre-fitted preprocessor.
    4. Selection of optimized features for inference.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame for inference, containing raw features that will undergo
        the same preprocessing steps as the training data.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame, including only the optimized features used during model training.
    """

    preprocessor: BaseEstimator = joblib.load(
        os.path.join(TRAINING_PREPROCESSOR, "fitted_preprocessor.pkl")
    )
    optimized_features: List[str] = joblib.load(
        os.path.join(OPTIMIZED_FEATURES, "optimized_features.pkl")
    )

    if "area" not in data.columns:
        data["area"] = "T"

    data_geo_encoded = geographical_encoding(data, spatial_features=["X", "Y"])
    data_binned = feature_binning(data_geo_encoded, features=["rain"])
    processed_array = preprocessor.transform(data_binned)
    transformed_column_names = preprocessor.get_feature_names_out()
    processed_inference_set = pd.DataFrame(
        processed_array,
        columns=transformed_column_names,
    )
    processed_data = processed_inference_set[optimized_features]
    processed_data.drop(columns=["area"], inplace=True, errors="ignore")
    return processed_data


def read_csv_from_minio(
    bucket_name: str,
    object_name: str,
) -> pd.DataFrame:
    """
    Read a CSV file from MinIO and load it into a pandas DataFrame.

    Parameters
    ----------
    bucket_name : str
        The name of the MinIO bucket where the CSV file is stored.
    object_name : str
        The name of the CSV file (object) in the MinIO bucket.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data from the CSV file.

    Raises
    -------
    NoSuchKey
        If the specified object does not exist in the MinIO bucket.
    Exception
        For any other errors that may occur while reading the CSV file.
    """

    try:
        client = Minio(
            MINIO_URL,
            access_key=MINIO_ROOT_USER,
            secret_key=MINIO_ROOT_PASSWORD,
            secure=False,
        )
        response = client.get_object(bucket_name, object_name)
        df = pd.read_csv(response)
        print(f"Data from '{object_name}':")
        return df
    except S3Error as e:
        print(
            f"Error: The object '{object_name}' does not exist in bucket '{bucket_name}' or another S3-related error occurred: {e}"
        )
        return None
    except MinioException as e:
        print(f"Error connecting to MinIO: {e}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def transform_and_validate_redis_data(
    redis_data: Dict[str, Union[int, float, str]],
) -> pd.DataFrame:
    """
    Converts a single JSON data entry retrieved from Redis into a Pandas DataFrame, validates the schema, and handles missing values or invalid domain values.

    Parameters
    ----------
    redis_data : dict
        The single row of data in JSON format retrieved from Redis.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with the correct schema validated and data types applied, missing values logged and rows skipped, domain compliance checked.
    """

    try:
        raw_data = read_csv_from_minio(
            bucket_name=MINIO_BUCKET,
            object_name=MINIO_OBJECT_NAME,
        )

        if raw_data is None:
            raise ValueError("Failed to retrieve data from MinIO.")

        schema = {
            column: raw_data[column].dtype
            for column in raw_data.columns
            if column != "area"
        }

        inference_data = pd.DataFrame([redis_data])
        redis_columns = set(inference_data.columns)
        schema_columns = set(schema.keys())

        if redis_columns != schema_columns:
            logger.error(
                f"Schema mismatch: Redis data columns {redis_columns} "
                f"do not match the expected schema columns {schema_columns}"
            )
            raise ValueError("Schema column mismatch.")

        domain_constraints = {
            "X": (1, 9),
            "Y": (2, 9),
            "month": [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ],
            "day": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
            "DMC": (1.1, 291.3),
            "FFMC": (18.7, 96.2),
            "DC": (7.9, 860.6),
            "ISI": (0.0, 56.1),
            "temp": (2.2, 33.3),
            "RH": (15.0, 100.0),
            "wind": (0.4, 9.4),
            "rain": (0.0, 6.4),
        }

        row = inference_data.iloc[0]
        if row.isnull().any():
            missing_cols = row[row.isnull()].index.tolist()
            logger.error(f"Row has missing values in columns: {missing_cols}.")
            raise ValueError("Missing values detected.")

        for column, dtype in schema.items():
            try:
                row[column] = (
                    pd.to_numeric(row[column], errors="raise")
                    if dtype.kind in ["i", "f"]
                    else row[column]
                )
            except ValueError as e:
                logger.error(f"Type casting error for column '{column}': {e}")
                raise ValueError(f"Type casting error for column '{column}'.")

        for column, constraint in domain_constraints.items():
            if isinstance(constraint, tuple):
                min_val, max_val = constraint
                if not (min_val <= row[column] <= max_val):
                    logger.error(
                        f"Feature '{column}' out of range: {row[column]}. "
                        f"Expected between {min_val} and {max_val}."
                    )
                    raise ValueError(f"Feature '{column}' out of range.")
            elif isinstance(constraint, list):
                if row[column] not in constraint:
                    logger.error(
                        f"Feature '{column}' has an invalid value: {row[column]}. "
                        f"Expected one of {constraint}."
                    )
                    raise ValueError(f"Invalid value in feature '{column}'.")

        return pd.DataFrame([row])

    except Exception as e:
        logger.exception("Error in transform_and_validate_redis_data")
        raise e


def load_data_or_schema(file_path: str) -> Tuple[pd.DataFrame, dict]:
    """
    Loads data from a CSV file into a Pandas DataFrame and returns the schema definition.

    Parameters
    ----------
    file_path : str
        The path to the CSV file that contains the data to be loaded.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the data from the specified CSV file.

    dict
        A dictionary where the keys are column names and the values are data types of the columns.
    """

    raw_data = pd.read_csv(file_path)
    schema = {column: raw_data[column].dtype for column in raw_data.columns}
    return raw_data, schema


def optimum_k(data: pd.DataFrame) -> Tuple[int, KMeans]:
    """
    Determines the optimal number of clusters (k) for KMeans clustering using the elbow method
    and silhouette scores. It evaluates k values in the range of 2 to 10.

    The function computes two metrics:
    1. **Inertia** (within-cluster sum of squares) for each k and identifies the elbow point,
       where the rate of decrease in inertia slows down, indicating the optimal k.
    2. **Silhouette Score** for each k, which measures how well clusters are separated.

    If the elbow method and silhouette score suggest the same k, that value is returned as the optimal k.
    Otherwise, the elbow method's result is returned with a message.

    Parameters
    -------
    data : pd.DataFrame
        The dataset to be clustered, with numerical features only.

    Returns
    -------
        Tuple containing the optimal number of clusters (k) if both methods agree, otherwise the elbow method's recommendation, and the fitted KMeans model corresponding to the optimal number of clusters.
    """

    inertia_values = []
    silhouette_scores = []
    fitted_models = []
    k_range = range(2, 11)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        inertia_values.append(km.inertia_)
        cluster_labels = km.labels_
        s_score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(s_score)
        fitted_models.append(km)

    inertia_diffs = np.diff(inertia_values)
    elbow_k = k_range[np.argmin(np.abs(inertia_diffs)) + 1]
    silhouette_k = k_range[np.argmax(silhouette_scores)]

    if silhouette_k == elbow_k:
        optimal_k = silhouette_k
    else:
        optimal_k = elbow_k

    optimal_model = fitted_models[k_range.index(optimal_k)]
    return optimal_k, optimal_model


def hybrid_cluster_outlier_removal(
    data: pd.DataFrame,
    kmeans_model: KMeans,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines KMeans for clustering and DBSCAN for outlier detection.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to be clustered and cleaned.
    kmeans_model : KMeans
        The fitted KMeans model from the optimum_k function.
    dbscan_eps : float
        The epsilon parameter for DBSCAN, representing the neighborhood size.
    dbscan_min_samples : int
        The minimum number of samples required to form a dense region in DBSCAN.

    Returns
    -------
        Tuple containing the cleaned and raw datasets
    """

    data["Cluster"] = kmeans_model.predict(data)
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(data.drop(columns=["Cluster"]))
    data["DBSCAN_Outlier"] = dbscan_labels == -1
    num_of_outliers = data["DBSCAN_Outlier"].sum()
    print(f"Number of data points: {len(data)}")
    print(f"Number of outliers detected: {num_of_outliers}")
    print(f"Percentage of outliers: {(num_of_outliers / len(data)) * 100:.2f}%")
    cleaned_df = (
        data[~data["DBSCAN_Outlier"]]
        .drop(columns=["DBSCAN_Outlier", "Cluster"])
        .reset_index(drop=True)
    )
    return cleaned_df, data


def embedded_feature_selection(
    data: pd.DataFrame,
    target: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Performs embedded feature selection using Logistic Regression and Random Forest classifiers.
    The method combines the most important features from both models to select optimal features.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset containing both features and the target variable.
    target : str
        The name of the target column (dependent variable) in the dataset.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - A DataFrame containing the selected important features along with the target column.
        - A list of the names of the selected important features.
    """

    features = data.columns[data.columns != target]
    X = data[features]
    y = data[target]
    logistic_regression_model = LogisticRegression(max_iter=1000, random_state=42)
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    logistic_regression_model.fit(X, y)
    random_forest_model.fit(X, y)
    logistic_regression_coefficients = np.abs(logistic_regression_model.coef_[0])
    random_forest_importances = random_forest_model.feature_importances_
    logistic_regression_coefficient_threshold = np.percentile(
        logistic_regression_coefficients, 50
    )
    random_forest_importance_threshold = np.percentile(random_forest_importances, 50)

    logistic_regression_model_important_coefficients = [
        feature
        for feature, coef in zip(X.columns, logistic_regression_coefficients)
        if coef > logistic_regression_coefficient_threshold
    ]

    random_forest_model_important_coefficients = [
        feature
        for feature, importance in zip(X.columns, random_forest_importances)
        if importance > random_forest_importance_threshold
    ]

    print(
        "Important Logistic Regression Features:",
        logistic_regression_model_important_coefficients,
    )
    print(
        "Important Random Forest Features:",
        random_forest_model_important_coefficients,
    )

    logistic_regression_coefficient_set = set(
        logistic_regression_model_important_coefficients
    )

    random_forest_coefficient_set = set(random_forest_model_important_coefficients)
    combined_coefficient_set = logistic_regression_coefficient_set.union(
        random_forest_coefficient_set
    )
    important_coefficients = list(combined_coefficient_set)
    feature_optimized_data = pd.concat([X[important_coefficients], y], axis=1)
    return feature_optimized_data, important_coefficients


def filter_feature_selection(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies filter-based feature selection by calculating the Variance Inflation Factor (VIF)
    for each feature and removing those with VIF greater than 10 to avoid multicollinearity.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing independent variables (features).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - A DataFrame showing the features and their corresponding VIF values for pruned features (VIF <= 10).
        - A DataFrame of the pruned features after multicollinearity has been removed.
    """

    vif_X = pd.DataFrame()
    vif_X["Feature"] = X.columns
    vif_X["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(len(X.columns))
    ]
    vif_pruned_X = vif_X[vif_X["VIF"] <= 10].reset_index(drop=True)
    vif_columns = vif_pruned_X["Feature"].tolist()
    pruned_X = X[vif_columns]
    return pruned_X, vif_columns


def custom_scoring(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Custom scoring function that computes a weighted score based on recall, F1 score,
    accuracy, and precision.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values returned by the classifier.

    Returns
    -------
    float
        A weighted score combining recall, F1 score, accuracy, and precision using the following weights:
        - Recall: 40%
        - F1 score: 30%
        - Accuracy: 20%
        - Precision: 10%
    """

    recall_weight = 0.4
    f1_weight = 0.3
    accuracy_weight = 0.2
    precision_weight = 0.1

    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    return (
        (recall_weight * recall)
        + (f1_weight * f1)
        + (accuracy_weight * accuracy)
        + (precision_weight * precision)
    )


def custom_scorer() -> make_scorer:
    """
    Wrapper function to create a custom scorer using the `custom_scoring` function.

    Returns
    -------
    scorer
        A scorer object to be used with GridSearchCV or other model evaluation tools.
        The scorer uses the custom scoring function and ensures that higher scores are better.
    """

    return make_scorer(custom_scoring, greater_is_better=True)


def binary_encoding(
    X: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """
    Encodes specified features with binary values, converting 'T' to 1 and 'F' to 0.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing features to encode.
    features : List[str]
        A list of column names (features) in the DataFrame to be encoded.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the specified features binary-encoded.
    """

    for feature in features:
        X[feature] = X[feature].replace({"T": 1, "F": 0})
    return X


def geographical_encoding(
    X: pd.DataFrame,
    spatial_features: List[str],
) -> pd.DataFrame:
    """
    Encodes geographical or spatial features by combining them into a single feature (e.g., grid zone).

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing spatial features.
    spatial_features : List[str]
        A list of spatial feature column names to combine.

    Returns
    -------
    pd.DataFrame
        The DataFrame with a new 'grid_zone' feature created by concatenating the specified spatial features.
        The original spatial features are dropped from the DataFrame.
    """

    X["grid_zone"] = X[spatial_features].astype(str).agg("_".join, axis=1)
    X.drop(columns=spatial_features, inplace=True)
    return X


def feature_binning(X: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Bins continuous features into binary values (1 or 0) based on whether the value is greater than 0.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing features to be binned.
    features : List[str]
        A list of continuous feature column names to apply binning.

    Returns
    -------
    pd.DataFrame
        The DataFrame with new binary-binned features added. Each binned feature is named as
        '{feature}_binned', where 'feature' is the original feature name.
    """

    for feature in features:
        X[f"{feature}_binned"] = X[feature].apply(lambda x: 1 if x > 0 else 0)
    X.drop(columns=features, inplace=True)
    return X


def hyperparameter_tuning(
    estimator: ClassifierMixin,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> ClassifierMixin:
    """
    Performs hyperparameter tuning for a given classification model using GridSearchCV.

    Parameters
    ----------
    estimator : ClassifierMixin
        The classification model to tune (e.g., LogisticRegression, DecisionTreeClassifier).
    param_grid : Dict[str, List[Any]]
        A dictionary containing hyperparameters and corresponding lists of values to search over.
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.DataFrame
        Target labels corresponding to X_train.

    Returns
    -------
    ClassifierMixin
        The best classifier model found by GridSearchCV after hyperparameter tuning.
    """

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=custom_scorer(),
        cv=RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=3,
            random_state=42,
        ),
        refit=True,
        n_jobs=6,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    tuned_model = grid_search.best_estimator_
    return tuned_model


def bayesian_hyperparameter_tuning(
    estimator: ClassifierMixin,
    search_spaces: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> ClassifierMixin:
    """
    Performs hyperparameter tuning for a given classification model using BayesSearchCV.

    Parameters
    ----------
    estimator : ClassifierMixin
        The classification model to tune (e.g., LogisticRegression, DecisionTreeClassifier).
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.

    Returns
    -------
    ClassifierMixin
        The best classifier model found by BayesSearchCV after hyperparameter tuning.
    """

    grid_search = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        scoring=custom_scorer(),
        cv=RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=3,
            random_state=42,
        ),
        refit=True,
        random_state=42,
        n_iter=50,
        n_jobs=6,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    tuned_model = grid_search.best_estimator_
    return tuned_model


def train_logistic_regression_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List[Any]],
) -> LogisticRegression:
    """
    Trains a Logistic Regression model with hyperparameter tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.
    param_grid : Dict[str, List[Any]]
        A dictionary of hyperparameters and their possible values for tuning.

    Returns
    -------
    LogisticRegression
        The best Logistic Regression model found after hyperparameter tuning.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Logistic Regression Classifier Experiment Run w/ Class Weights - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Forest Fire Logistic Regression Classifier"

    with mlflow.start_run(run_name=model_name) as run:
        logistic_regression_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight={0: 1, 1: 5},
        )

        logistic_regression_tuned_model = hyperparameter_tuning(
            estimator=logistic_regression_model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
        )

        logistic_regression_best_params = logistic_regression_tuned_model.get_params()
        logistic_regression_tuned_model.fit(X_train, y_train)
        logistic_regression_pred = logistic_regression_tuned_model.predict(X_train)
        accuracy = accuracy_score(y_train, logistic_regression_pred)
        precision = precision_score(y_train, logistic_regression_pred)
        recall = recall_score(y_train, logistic_regression_pred)
        f1 = f1_score(y_train, logistic_regression_pred)
        mlflow.sklearn.log_model(logistic_regression_tuned_model, model_name)
        mlflow.log_params(logistic_regression_best_params)
        mlflow.log_metrics(
            {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

        try:
            model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
            model_description = (
                f"Logistic Regression Classifier for Forest Fire Predictions with "
                f"best parameters: {logistic_regression_best_params}. "
            )

            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

    return logistic_regression_tuned_model


def train_decision_tree_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List[Any]],
) -> DecisionTreeClassifier:
    """
    Trains a Decision Tree classifier with hyperparameter tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.
    param_grid : Dict[str, List[Any]]
        A dictionary of hyperparameters and their possible values for tuning.

    Returns
    -------
    DecisionTreeClassifier
        The best Decision Tree classifier found after hyperparameter tuning.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name: str = (
        f"Decision Tree Classifier Experiment Run - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Forest Fire Decision Tree Classifier"

    with mlflow.start_run(run_name=model_name) as run:
        decision_tree_model = DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        )

        decision_tree_tuned_model = hyperparameter_tuning(
            estimator=decision_tree_model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
        )

        decision_tree_best_params = decision_tree_tuned_model.get_params()
        decision_tree_tuned_model.fit(X_train, y_train)
        decision_tree_pred = decision_tree_tuned_model.predict(X_train)
        accuracy = accuracy_score(y_train, decision_tree_pred)
        precision = precision_score(y_train, decision_tree_pred)
        recall = recall_score(y_train, decision_tree_pred)
        f1 = f1_score(y_train, decision_tree_pred)
        mlflow.sklearn.log_model(decision_tree_tuned_model, model_name)
        mlflow.log_params(decision_tree_best_params)
        mlflow.log_metrics(
            {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

        try:
            model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
            model_description = (
                f"Decision Tree Classifier for Forest Fire Predictions with "
                f"best parameters: {decision_tree_best_params}. "
            )

            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

    return decision_tree_tuned_model


def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_spaces: Dict[str, Any],
) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier with Bayesian hyperparameter tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.
    search_spaces : Dict[str, Any]
        A dictionary of hyperparameters and their possible values for tuning.

    Returns
    -------
    RandomForestClassifier
        The best Random Forest classifier found by BayesSearchCV after hyperparameter tuning.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Random Forest Classifier Experiment Run - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Forest Fire Random Forest Classifier"

    with mlflow.start_run(run_name=model_name) as run:
        random_forest_model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
        )

        random_forest_tuned_model = bayesian_hyperparameter_tuning(
            estimator=random_forest_model,
            search_spaces=search_spaces,
            X_train=X_train,
            y_train=y_train,
        )

        random_forest_best_params = random_forest_tuned_model.get_params()
        random_forest_tuned_model.fit(X_train, y_train)
        random_forest_pred = random_forest_tuned_model.predict(X_train)
        accuracy = accuracy_score(y_train, random_forest_pred)
        precision = precision_score(y_train, random_forest_pred)
        recall = recall_score(y_train, random_forest_pred)
        f1 = f1_score(y_train, random_forest_pred)
        mlflow.sklearn.log_model(random_forest_tuned_model, model_name)
        mlflow.log_params(random_forest_best_params)
        mlflow.log_metrics(
            {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

        try:
            model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
            model_description = (
                f"Random Forest Classifier for Forest Fire Predictions with "
                f"best parameters: {random_forest_best_params}. "
            )

            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

    return random_forest_tuned_model


def train_mlp_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List[Any]],
) -> MLPClassifier:
    """
    Trains a Multi-Layer Perceptron (MLP) classifier with hyperparameter tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.
    param_grid : Dict[str, List[Any]]
        A dictionary of hyperparameters and their possible values for tuning.

    Returns
    -------
    MLPClassifier
        The best MLP classifier found after hyperparameter tuning.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Multi-Layer Perceptron Classifier Experiment Run - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Forest Fire MLP Classifier"

    with mlflow.start_run(run_name=model_name) as run:
        mlp_model = MLPClassifier(
            random_state=42,
            max_iter=2000,
            early_stopping=True,
        )

        mlp_tuned_model = hyperparameter_tuning(
            estimator=mlp_model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
        )

        mlp_best_params = mlp_tuned_model.get_params()
        mlp_tuned_model.fit(X_train, y_train)
        mlp_pred = mlp_tuned_model.predict(X_train)
        accuracy = accuracy_score(y_train, mlp_pred)
        precision = precision_score(y_train, mlp_pred)
        recall = recall_score(y_train, mlp_pred)
        f1 = f1_score(y_train, mlp_pred)
        mlflow.sklearn.log_model(mlp_tuned_model, model_name)
        mlflow.log_params(mlp_best_params)
        mlflow.log_metrics(
            {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

        try:
            model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
            model_description = (
                f"MLP Classifier for Forest Fire Predictions with "
                f"best parameters: {mlp_best_params}. "
            )

            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            print(f"Model registration for {model_name} failed: {e}")

    return mlp_tuned_model


def load_trained_model() -> Tuple[
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    MLPClassifier,
]:
    """
    Loads the trained models (Logistic Regression, Decision Tree, Random Forest, MLP).

    Returns
    -------
    tuple: A tuple containing:
        - LogisticRegression model
        - DecisionTreeClassifier model
        - RandomForestClassifier model
        - MLPClassifier model
    """

    logistic_regression_model = joblib.load(TRAINED_LOGISTIC_REGRESSION_CLASSIFIER)
    decision_tree_model = joblib.load(TRAINED_DECISION_TREE_CLASSIFIER)
    random_forest_model = joblib.load(TRAINED_RANDOM_FOREST_CLASSIFIER)
    mlp_model = joblib.load(TRAINED_MLP_CLASSIFIER)

    return (
        logistic_regression_model,
        decision_tree_model,
        random_forest_model,
        mlp_model,
    )


def load_trained_model_by_name_and_version(
    model_name: str,
    version: int,
) -> PyFuncModel:
    """
    Loads a model from the MLflow registry based on the specified name and version.

    Parameters
    ----------
    model_name : str
        The registered name of the model in MLflow.
    version : int
        The specific version of the model to load.

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        The MLflow model ready for predictions.
    """

    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def model_validation(
    estimator: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimator_name: str,
) -> pd.DataFrame:
    """
    Perform cross-validation on a given classifier to evaluate its performance.

    This function uses Stratified K-Fold cross-validation to assess the performance of the provided estimator
    on the training data. The cross-validation scores are computed using a custom scoring function.

    Parameters
    ----------
    estimator : (ClassifierMixin)
        The classifier to be evaluated. This should be an instance of a scikit-learn classifier that implements the `fit` and `predict` methods.
    X_train : (pd.DataFrame)
        The feature set used for training the estimator. It should be a DataFrame with shape (n_samples, n_features).
    y_train : (pd.Series)
        The target labels corresponding to the training features. It should be a Series with shape (n_samples,).
    estimator_name : (str)
        A string representing the name of the estimator. This is used for labeling the results in the output DataFrame.

    Returns
    -------
        pd.DataFrame: A DataFrame containing the cross-validation scores for each fold, with the
        model name as the first column. The DataFrame has the following structure:
    """

    stratified_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    cross_val_scores = cross_val_score(
        estimator, X_train, y_train, cv=stratified_cv, scoring=custom_scorer()
    )
    results = pd.DataFrame(
        data=[cross_val_scores],
        columns=["1st fold", "2nd fold", "3rd fold", "4th fold", "5th fold"],
    )
    results.insert(0, "Model", estimator_name)
    return results


def model_prediction(
    estimator: ClassifierMixin,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Makes predictions for a binary classification model based on a custom decision threshold.

    Parameters
    ----------
    estimator : ClassifierMixin
        A trained classification model that has the `predict_proba` method for predicting probabilities, or a PyFuncModel which has only `predict`.
    X_test : pd.DataFrame
        Test data containing feature variables for which predictions need to be made.

    Returns
    -------
    np.ndarray
        A NumPy array of binary predictions (0 or 1), where the decision threshold is applied to the predicted probabilities (if available).

    Notes
    ------
    - The method uses `predict_proba` to obtain predicted probabilities for the positive class (class 1).
    - A custom decision threshold of 0.4 is applied to determine the final binary prediction.
    - Predictions with a probability greater than or equal to 0.4 are classified as class 1, otherwise class 0.
    """

    if hasattr(estimator, "predict_proba"):
        y_pred_prob = estimator.predict_proba(X_test)[:, 1]
    else:
        y_pred_prob = estimator.predict(X_test)

        if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 2:
            y_pred_prob = y_pred_prob[:, 1]
        elif y_pred_prob.ndim == 1:
            return y_pred_prob

    decision_threshold = 0.4
    y_pred = (y_pred_prob >= decision_threshold).astype(int)
    return y_pred


def model_confusion_matrix(
    estimator: ClassifierMixin,
    y_test: pd.Series,
    y_pred: np.ndarray,
    ax: plt.Axes,
) -> None:
    """
    Plots the confusion matrix for a given model's predictions.

    Parameters
    ----------
    model : ClassifierMixin
        The trained classification model which has the `classes_` attribute representing the class labels.
    y_test : pd.Series
        The true labels for the test dataset.
    y_pred : np.ndarray
        The predicted labels generated by the model.
    ax : plt.Axes
        The matplotlib Axes on which to plot the confusion matrix.

    Returns
    -------
    None
        This function doesn't return any value but displays the confusion matrix plot on the provided Axes.

    Notes
    ------
    - The confusion matrix is computed using the true and predicted labels (`y_test` and `y_pred`).
    - The model's `classes_` attribute is used to ensure the correct order of class labels.
    - The confusion matrix is displayed using `ConfusionMatrixDisplay` for clearer visualization.
    """

    cm = confusion_matrix(y_test, y_pred, labels=estimator.classes_)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=estimator.classes_)
    cm_display.plot(ax=ax)


def model_classification_report(
    estimator_predictions: Dict[str, Any],
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Generates a classification report for multiple models and returns the results in a DataFrame.

    Parameters
    ----------
    estimator_predictions : Dict[str, Any]
        A dictionary where the keys are model names (strings) and the values are the predicted labels (np.ndarray) for the test set from each model.
    y_test : pd.Series
        The true labels for the test dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing model performance metrics for each model, including Accuracy, Precision, Recall, and F1 Score for the positive class (`1.0`).

    Notes
    ------
    - The function iterates over the dictionary of model predictions, computes the classification report for each model, and stores key metrics.
    - The output includes the Accuracy, Precision, Recall, and F1 Score specifically for the positive class (`1.0`).
    - The function returns a pandas DataFrame summarizing these metrics for all the models.
    """

    classification_report_data: List[Dict[str, Any]] = []
    for model_name, model_pred in estimator_predictions.items():
        report = classification_report(y_test, model_pred, output_dict=True)
        classification_report_data.append(
            {
                "Model": model_name,
                "Accuracy Score": report["accuracy"],
                "Precision Score": report["1.0"]["precision"],
                "Recall Score": report["1.0"]["recall"],
                "F1 Score": report["1.0"]["f1-score"],
            }
        )

    classification_report_df = pd.DataFrame(classification_report_data)
    return classification_report_df
