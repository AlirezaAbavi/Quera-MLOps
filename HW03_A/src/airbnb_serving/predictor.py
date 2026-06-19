from typing import Iterable

import numpy as np
import pandas as pd

from .schema import ListingFeatures, PredictionResponse


def _get_expected_columns(model) -> list[str] | None:
    """Infer the feature columns expected by the trained sklearn model/pipeline."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def _get_positive_class_index(model) -> int:
    """Find the probability column for positive class 1."""
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                classes = step.classes_
                break

    if classes is None:
        return 1

    classes = list(classes)
    if 1 in classes:
        return classes.index(1)

    return len(classes) - 1


def _features_to_dataframe(features_list: Iterable[ListingFeatures], model) -> pd.DataFrame:
    """Convert validated Pydantic inputs into the model input DataFrame."""
    rows = [item.model_dump() for item in features_list]

    if not rows:
        raise ValueError("At least one listing is required for prediction.")

    df = pd.DataFrame(rows)

    # Public API field -> model training field.
    if "host_is_superhost" in df.columns and "is_superhost" not in df.columns:
        df["is_superhost"] = df["host_is_superhost"].astype(bool)

    # Internal derived feature expected by the trained model.
    if "has_reviews_before_cutoff" not in df.columns:
        df["has_reviews_before_cutoff"] = (
            df["total_reviews_before_cutoff"].fillna(0).astype(float) > 0
        )

    expected_columns = _get_expected_columns(model)

    if expected_columns is not None:
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing model input columns: {missing_columns}")

        return df[expected_columns]

    return df


def _predict_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities if available; otherwise use predictions."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        positive_idx = _get_positive_class_index(model)
        return proba[:, positive_idx]

    predictions = model.predict(X)
    return predictions.astype(float)


def predict_single(
    features: ListingFeatures,
    model,
    run_id: str,
) -> PredictionResponse:
    """Predict high-demand probability for one listing."""
    X = _features_to_dataframe([features], model)

    predictions = model.predict(X)
    probabilities = _predict_probabilities(model, X)

    prediction = int(predictions[0])
    probability = float(probabilities[0])

    return PredictionResponse(
        listing_id=None,
        prediction=prediction,
        probability_high_demand=probability,
        model_run_id=str(run_id),
    )


def predict_batch(
    features_list: list[ListingFeatures],
    model,
    run_id: str,
) -> list[PredictionResponse]:
    """Predict high-demand probability for a batch of listings.

    The model is called once for the full batch.
    """
    X = _features_to_dataframe(features_list, model)

    predictions = model.predict(X)
    probabilities = _predict_probabilities(model, X)

    responses: list[PredictionResponse] = []

    for prediction, probability in zip(predictions, probabilities):
        responses.append(
            PredictionResponse(
                listing_id=None,
                prediction=int(prediction),
                probability_high_demand=float(probability),
                model_run_id=str(run_id),
            )
        )

    return responses
