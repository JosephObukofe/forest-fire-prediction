import asyncio
import logging
import redis.asyncio as redis
from celery import Celery
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from contextlib import asynccontextmanager
from pydantic import ValidationError
from app.schema import InputData
from app.pubsub import publish
from predicting_forest_fires.config.config import (
    REDIS_HOST_NAME,
    REDIS_PORT,
    REDIS_DB_PASSWORD,
)
from app.predict import predict


app = FastAPI()
redis_url = f"redis://:{REDIS_DB_PASSWORD}@{REDIS_HOST_NAME}:{REDIS_PORT}"
celery_app = Celery(
    "predicting-forest-fires",
    broker=redis_url,
    backend=redis_url,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = redis.from_url(redis_url)
    app.state.redis_client = redis_client
    yield
    await redis_client.close()


app = FastAPI(lifespan=lifespan)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)


@app.post("/submit_input/")
async def submit_input(data: InputData):
    try:
        logger.debug(f"Received data for prediction: {data.model_dump()}")
        await publish(
            channel="inference",
            message=data.model_dump(),
        )
        task = celery_app.send_task(
            "app.predict.predict",
            args=[data.model_dump()],
        )
        logger.info(f"Task ID {task.id} dispatched for prediction.")
        return {
            "task_id": task.id,
            "status": "Predicting",
        }
    except ValidationError as e:
        logger.error("Validation error: %s", e)
        raise HTTPException(
            status_code=422,
            detail="Validation Error: " + str(e),
        )
    except Exception as e:
        logger.error("Unexpected error in submit_input: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Server Error: " + str(e),
        )


@app.websocket("/ws/prediction/")
async def prediction_websocket(websocket: WebSocket):
    await websocket.accept()
    redis_client = app.state.redis_client
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("results")

    try:
        logger.info("WebSocket connection established")
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True)
            if message and message["type"] == "message":
                prediction_result = message["data"]
                await websocket.send_text(prediction_result.decode())
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("Error in WebSocket connection: %s", e)
    finally:
        await pubsub.unsubscribe("results")
        await pubsub.close()
        if not websocket.client_state.closed:
            await websocket.close()
        logger.info("WebSocket connection closed")
