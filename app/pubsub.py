import redis
import logging
from datetime import time
from predicting_forest_fires.config.config import (
    REDIS_HOST_NAME,
    REDIS_PORT,
    REDIS_DB_PASSWORD,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = redis.Redis(
    host=REDIS_HOST_NAME,
    port=REDIS_PORT,
    password=REDIS_DB_PASSWORD,
    decode_responses=True,
)


def publish(channel: str, message: str):
    """
    Publishes a message to a Redis channel.
    """

    try:
        r.publish(channel, message)
        logger.info(f"Published message to {channel}: {message}")
    except Exception as e:
        logger.error(f"Failed to publish message to {channel}: {str(e)}")
        raise e
