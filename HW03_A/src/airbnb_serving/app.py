import os
from pathlib import Path
from typing import List

import joblib
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, status

from .predictor import predict_batch, predict_single
from .schema import ListingFeatures, PredictionResponse


DEFAULT_RUN_ID = "2f4885cc6b2b42baa62c279937ea4401"
DEFAULT_EXPERIMENT_NAME = "qbc12_hw02_student_alireza_abouei"
DEFAULT_LOCAL_MODEL_PATH = "mlflow_artifacts/v5_random_forest/model.joblib"


RUN_ID = os.getenv("MLFLOW_RUN_ID", DEFAULT_RUN_ID)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://185.50.38.163:33014")
MODEL_URI = os.getenv("MODEL_URI", f"runs:/{RUN_ID}/model")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", DEFAULT_LOCAL_MODEL_PATH)

model = None
model_load_mode = None
model_load_error = None


def load_model():
    """Load the selected model.

    First tries MLflow online loading. If that fails, it falls back to a local joblib
    model artifact. The local fallback is useful when the MLflow artifact server is
    slow or temporarily returns 500 errors.
    """
    global model, model_load_mode, model_load_error

    if model is not None:
        return model

    mlflow.set_tracking_uri(TRACKING_URI)

    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        model_load_mode = "mlflow_online"
        model_load_error = None
        return model

    except Exception as online_error:
        local_path = Path(LOCAL_MODEL_PATH)

        if not local_path.exists():
            model_load_error = (
                "MLflow loading failed and local fallback model was not found. "
                f"MLflow error: {online_error!r}. "
                f"Local path tried: {local_path}"
            )
            raise RuntimeError(model_load_error) from online_error

        try:
            model = joblib.load(local_path)
            model_load_mode = "local_joblib_fallback"
            model_load_error = None
            return model

        except Exception as local_error:
            model_load_error = (
                "Both MLflow loading and local joblib fallback loading failed. "
                f"MLflow error: {online_error!r}. "
                f"Local error: {local_error!r}"
            )
            raise RuntimeError(model_load_error) from local_error


app = FastAPI(
    title="Airbnb Demand Prediction API",
    description="Serve the selected HW02 classic ML model with FastAPI.",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    """Try loading the model during startup.

    If loading fails, the API still starts so /health can show the error.
    """
    try:
        load_model()
    except Exception:
        # Keep the service alive. /health will report the loading problem.
        pass


@app.get("/")
def root():
    return {
        "service": "airbnb-demand-prediction",
        "status": "running",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "model_load_mode": model_load_mode,
        "model_run_id": RUN_ID,
        "model_uri": MODEL_URI,
        "error": model_load_error,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_run_id": RUN_ID,
        "experiment_name": EXPERIMENT_NAME,
        "model_uri": MODEL_URI,
        "tracking_uri": TRACKING_URI,
        "model_load_mode": model_load_mode,
        "local_model_path": LOCAL_MODEL_PATH,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: ListingFeatures):
    try:
        loaded_model = load_model()
        return predict_single(features, loaded_model, RUN_ID)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/predict-batch", response_model=List[PredictionResponse])
def predict_batch_endpoint(features: List[ListingFeatures]):
    if not features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch request must contain at least one listing.",
        )

    try:
        loaded_model = load_model()
        return predict_batch(features, loaded_model, RUN_ID)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        ) from exc
