import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from predicting_forest_fires.data.custom import (
    read_csv_from_minio,
    optimum_k,
    hybrid_cluster_outlier_removal,
    embedded_feature_selection,
    filter_feature_selection,
    binary_encoding,
    geographical_encoding,
    feature_binning,
    preprocess_training_set,
    preprocess_test_set,
)
from predicting_forest_fires.delta.setup import get_spark_session, write_to_delta
from predicting_forest_fires.config.config import (
    MODEL_TRAINING_DATA,
    MODEL_VALIDATION_DATA,
    EVALUATION_TEST_DATA,
    TRAINING_PREPROCESSOR,
    OPTIMIZED_FEATURES,
    DELTA_TRAINING_PATH,
    DELTA_TEST_PATH,
    MINIO_BUCKET,
    MINIO_OBJECT_NAME,
)

df = read_csv_from_minio(
    bucket_name=MINIO_BUCKET,
    object_name=MINIO_OBJECT_NAME,
)

df_encoded = binary_encoding(df, features=["area"])
df_geo_encoded = geographical_encoding(df_encoded, spatial_features=["X", "Y"])
df_binned = feature_binning(df_geo_encoded, features=["rain"])

target = "area"
features = df_binned.columns[df_binned.columns != target]
X = df_binned[features]
y = df_binned[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Feature Categorization
log_transform_features = ["ISI"]
modified_log_transform_features = ["FFMC"]
numerical_features = [
    feature
    for feature in list(df_binned.select_dtypes(exclude=object).columns)
    if feature not in ["area", "rain_binned"]
]
categorical_features = list(df_binned.select_dtypes(include=object).columns)

# Preprocessing Training Set
X_train_processed, preprocessor = preprocess_training_set(
    pd.concat([X_train, y_train], axis=1),
    log_transform_features=log_transform_features,
    modified_log_transform_features=modified_log_transform_features,
    numerical_features=numerical_features,
    categorical_features=categorical_features,
)

# Preprocessing Test Set
X_test_processed = preprocess_test_set(
    pd.concat([X_test, y_test], axis=1),
    preprocessor=preprocessor,
)

# Outlier Detection and Removal
_, kmeans_model = optimum_k(data=X_train_processed)
X_train_outliers, _ = hybrid_cluster_outlier_removal(
    data=X_train_processed,
    kmeans_model=kmeans_model,
    dbscan_eps=3.5,
    dbscan_min_samples=2,
)
X_train_indices = X_train_outliers.index
X_train_trimmed = X_train_outliers.loc[X_train_indices].reset_index(drop=True)

# Embedded Feature Selection
X_train_embedded_reduced, _ = embedded_feature_selection(X_train_trimmed, target="area")

# Filter Feature Selection
X_train_filter_reduced, optimized_features = filter_feature_selection(X_train_embedded_reduced)

# Feature Selection for Test Set
X_test_reduced = X_test_processed[optimized_features]

# Save the preprocessed training, test datasets, and the training preprocessor to disk using joblib
joblib.dump(X_train_filter_reduced, MODEL_TRAINING_DATA + "/preprocessed_training_set.pkl")
joblib.dump(X_train_filter_reduced, MODEL_VALIDATION_DATA + "/preprocessed_training_set.pkl")
joblib.dump(X_test_reduced, EVALUATION_TEST_DATA + "/preprocessed_test_set.pkl")
joblib.dump(preprocessor, TRAINING_PREPROCESSOR + "/fitted_preprocessor.pkl")
joblib.dump(optimized_features, OPTIMIZED_FEATURES + "/optimized_features.pkl")

# Write the preprocessed training and test data to Delta Lake
spark = get_spark_session()
write_to_delta(spark, X_train_filter_reduced, DELTA_TRAINING_PATH)
write_to_delta(spark, X_test_reduced, DELTA_TEST_PATH)
