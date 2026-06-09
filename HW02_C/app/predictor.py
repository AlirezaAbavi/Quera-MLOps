from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException, status

from . import config
from .schemas import ListingFeatures, PredictionResponse


def records_to_dataframe(records: Iterable[ListingFeatures]) -> pd.DataFrame:
    """Convert validated API payloads into the exact DataFrame expected by the model."""
    rows = [record.model_dump() for record in records]
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "At least one record is required."},
        )

    df = pd.DataFrame(rows)

    forbidden_fields = sorted(set(df.columns) & set(config.FORBIDDEN_FIELDS))
    if forbidden_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Request contains forbidden leakage/audit/target fields.",
                "forbidden_fields": forbidden_fields,
            },
        )

    api_feature_columns = getattr(config, "API_FEATURE_COLUMNS", config.EXPECTED_FEATURE_COLUMNS)

    unexpected_fields = sorted(set(df.columns) - set(api_feature_columns))
    if unexpected_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Request contains fields that are not accepted API inputs.",
                "unexpected_fields": unexpected_fields,
            },
        )

    missing_api_fields = [col for col in api_feature_columns if col not in df.columns]
    if missing_api_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Request is missing required API input fields.",
                "missing_fields": missing_api_fields,
            },
        )

    # API field -> model field
    df["is_superhost"] = df["host_is_superhost"].astype(bool)

    # Derived internal model feature
    df["has_reviews_before_cutoff"] = (
        df["total_reviews_before_cutoff"].fillna(0).astype(float) > 0
    )

    missing_model_fields = [col for col in config.EXPECTED_FEATURE_COLUMNS if col not in df.columns]
    if missing_model_fields:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Internal model feature mapping is incomplete.",
                "missing_fields": missing_model_fields,
            },
        )

    return df[config.EXPECTED_FEATURE_COLUMNS]


def _positive_class_index(model) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return 1

    classes_list = list(classes)
    if 1 in classes_list:
        return classes_list.index(1)
    if True in classes_list:
        return classes_list.index(True)
    return min(1, len(classes_list) - 1)


def _probabilities_or_none(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return None

    positive_index = _positive_class_index(model)
    return probabilities[:, positive_index]


def predict_records(model, records: List[ListingFeatures]) -> List[PredictionResponse]:
    """Run model prediction and return clean API responses."""
    X = records_to_dataframe(records)

    try:
        probabilities = _probabilities_or_none(model, X)
        if probabilities is not None:
            predictions = (probabilities >= config.PREDICTION_THRESHOLD).astype(int)
        else:
            raw_predictions = model.predict(X)
            predictions = np.asarray(raw_predictions).astype(int)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    responses: List[PredictionResponse] = []
    for index, prediction in enumerate(predictions):
        prediction_int = int(prediction)
        probability = None if probabilities is None else round(float(probabilities[index]), 6)
        responses.append(
            PredictionResponse(
                prediction=prediction_int,
                prediction_label=config.POSITIVE_LABEL if prediction_int == 1 else config.NEGATIVE_LABEL,
                probability=probability,
                threshold=config.PREDICTION_THRESHOLD,
            )
        )

    return responses
