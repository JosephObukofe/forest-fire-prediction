import logging
from celery import shared_task
from app.pubsub import publish
from predicting_forest_fires.data.custom import (
    load_trained_model_by_name_and_version,
    transform_and_validate_redis_data,
    preprocess_inference_set,
    model_prediction,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@shared_task
def predict(data):
    """
    Processes input data, makes a prediction, and returns the result.
    """

    try:
        logger.info(f"Starting prediction task with data: {data}")
        validated_data = transform_and_validate_redis_data(redis_data=data)
        logger.info(f"Data validated successfully: {validated_data}")
        processed_data = preprocess_inference_set(data=validated_data)
        logger.info(f"Data processed successfully: {processed_data}")
        model = load_trained_model_by_name_and_version(
            model_name="Forest Fire Random Forest Classifier",
            version=2,
        )
        logger.info("Model loaded successfully")
        prediction = model_prediction(
            estimator=model,
            X_test=processed_data,
        )
        results = "".join(["No" if pred == 0 else "Yes" for pred in prediction])
        logger.info(f"Prediction results generated: {results}")
        publish(channel="results", message=results)
        logger.info("Prediction results published to Redis channel 'results'")
        return results
    except Exception as e:
        logger.error("Prediction task failed: %s", str(e))
        raise e
