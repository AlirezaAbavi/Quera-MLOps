from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

# Load local .env values before reading settings.
load_dotenv()

APP_TITLE = "QBC12 HW02 Listing Availability Prediction API"
APP_VERSION = "1.0.0"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://185.50.38.163:33014")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
STUDENT_USERNAME = os.getenv("STUDENT_USERNAME", MLFLOW_TRACKING_USERNAME or "student_alireza_abouei")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", f"qbc12_hw02_{STUDENT_USERNAME}")

# Final clean HW02 run selected in the experiment notebook.
# This can be overridden with an environment variable when testing another run.
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID", "2f4885cc6b2b42baa62c279937ea4401").strip()
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

# Startup should not block Swagger if MLflow is slow/unreachable.
# The model is loaded in a background thread and also retried on prediction if needed.
LOAD_MODEL_ON_STARTUP = os.getenv("LOAD_MODEL_ON_STARTUP", "true").strip().lower() in {"1", "true", "yes", "y"}
MLFLOW_HTTP_REQUEST_TIMEOUT = os.getenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", MLFLOW_HTTP_REQUEST_TIMEOUT)

DATASET_VERSION = "v1_student"
TARGET_NAME = "high_demand_proxy"
POSITIVE_LABEL = "high_demand_proxy"
NEGATIVE_LABEL = "not_high_demand_proxy"

API_FEATURE_COLUMNS: List[str] = [
    "room_type",
    "property_type",
    "neighbourhood_name",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "listing_price",
    "minimum_nights",
    "maximum_nights",
    "instant_bookable",
    "host_is_superhost",
    "host_listing_count",
    "total_reviews_before_cutoff",
    "unique_reviewers_before_cutoff",
    "avg_comment_len_before_cutoff",
    "max_comment_len_before_cutoff",
    "days_since_last_review",
    "available_days_last_90d",
    "available_rate_last_90d",
    "avg_minimum_nights_calendar_last_90d",
    "avg_maximum_nights_calendar_last_90d",
    "available_days_last_30d",
    "available_rate_last_30d",
    "avg_minimum_nights_calendar_last_30d",
    "avg_maximum_nights_calendar_last_30d",
]

EXPECTED_FEATURE_COLUMNS: List[str] = [
    "room_type",
    "property_type",
    "neighbourhood_name",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "listing_price",
    "minimum_nights",
    "maximum_nights",
    "instant_bookable",
    "is_superhost",
    "host_listing_count",
    "total_reviews_before_cutoff",
    "unique_reviewers_before_cutoff",
    "avg_comment_len_before_cutoff",
    "max_comment_len_before_cutoff",
    "days_since_last_review",
    "available_days_last_90d",
    "available_rate_last_90d",
    "avg_minimum_nights_calendar_last_90d",
    "avg_maximum_nights_calendar_last_90d",
    "available_days_last_30d",
    "available_rate_last_30d",
    "avg_minimum_nights_calendar_last_30d",
    "avg_maximum_nights_calendar_last_30d",
    "has_reviews_before_cutoff",
]

FORBIDDEN_FIELDS: List[str] = [
    "listing_id",
    "cutoff_date",
    "dataset_version",
    "future_calendar_days_observed_30d",
    "future_available_days_30d",
    "future_available_rate_30d",
    "high_demand_proxy",
]
