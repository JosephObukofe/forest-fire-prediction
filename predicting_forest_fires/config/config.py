import os
from dotenv import load_dotenv

load_dotenv()

RAW_DATA = os.getenv("RAW_DATA")
MODEL_TRAINING_DATA = os.getenv("MODEL_TRAINING_DATA")
MODEL_VALIDATION_DATA = os.getenv("MODEL_VALIDATION_DATA")
EVALUATION_TEST_DATA = os.getenv("EVALUATION_TEST_DATA")
TRAINING_PREPROCESSOR = os.getenv("TRAINING_PREPROCESSOR")
OPTIMIZED_FEATURES = os.getenv("OPTIMIZED_FEATURES")
TRAINED_LOGISTIC_REGRESSION_CLASSIFIER = os.getenv("TRAINED_LOGISTIC_REGRESSION_CLASSIFIER")
TRAINED_DECISION_TREE_CLASSIFIER = os.getenv("TRAINED_DECISION_TREE_CLASSIFIER")
TRAINED_RANDOM_FOREST_CLASSIFIER = os.getenv("TRAINED_RANDOM_FOREST_CLASSIFIER")
TRAINED_MLP_CLASSIFIER = os.getenv("TRAINED_MLP_CLASSIFIER")
CONFUSION_MATRIX = os.getenv("CONFUSION_MATRIX")
MODEL_EVALUATION_REPORT = os.getenv("MODEL_EVALUATION_REPORT")
MODEL_VALIDATION_REPORT = os.getenv("MODEL_VALIDATION_REPORT")
MINIO_URL = os.getenv("MINIO_URL")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")
MINIO_PATH = os.getenv("MINIO_PATH")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")
MINIO_OBJECT_NAME = os.getenv("MINIO_OBJECT_NAME")
DELTA_TRAINING_PATH = os.getenv("DELTA_TRAINING_PATH")
DELTA_TEST_PATH = os.getenv("DELTA_TEST_PATH")
REDIS_HOST_NAME = os.getenv("REDIS_HOST_NAME")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB_PASSWORD = os.getenv("REDIS_DB_PASSWORD")
FAST_API_BASE_URL = os.getenv("FAST_API_BASE_URL")
WEB_SOCKET_URL = os.getenv("WEB_SOCKET_URL")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_BACKEND_STORE_URI = os.getenv("MLFLOW_BACKEND_STORE_URI")
MLFLOW_DEFAULT_ARTIFACT_ROOT = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")