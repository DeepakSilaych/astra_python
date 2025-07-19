import os
from celery import Celery

# Initialize Celery
celery_app = Celery(
    "purecv_tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["tasks"],  # auto-discover tasks module
)

# Use JSON serialization for tasks and results
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,  # expire task results after 1 hour
    broker_transport_options={
        "visibility_timeout": 3600
    },  # visibility timeout in seconds
)
