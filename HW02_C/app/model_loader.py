from __future__ import annotations

import os
import tempfile
from threading import Lock
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import config


@dataclass
class LoadedModelState:
    model: Any = None
    loaded: bool = False
    error: Optional[str] = None
    model_uri: Optional[str] = None
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)


class ModelService:
    """Load the selected HW02 model from MLflow.

    The HW02 notebook may have logged either a native MLflow sklearn model or,
    when the tracking server did not support the newer logged-model API, a
    fallback joblib artifact under ``model_fallback/model.joblib``. This loader
    supports both formats so the FastAPI service can run reliably.
    """

    def __init__(self) -> None:
        self.state = LoadedModelState()
        self._load_lock = Lock()

    def _configure_mlflow(self):
        import mlflow
        from mlflow.tracking import MlflowClient

        if config.MLFLOW_TRACKING_USERNAME:
            os.environ["MLFLOW_TRACKING_USERNAME"] = config.MLFLOW_TRACKING_USERNAME
        if config.MLFLOW_TRACKING_PASSWORD:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        return MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)

    @staticmethod
    def _is_clean_finished_run(run) -> bool:
        tags = run.data.tags
        metrics = run.data.metrics
        run_name = tags.get("mlflow.runName", "")
        leakage_status = tags.get("leakage_status", "")

        if run.info.status != "FINISHED":
            return False
        if leakage_status == "leaky" or "leaky" in run_name.lower():
            return False
        if "f1" not in metrics or "roc_auc" not in metrics:
            return False
        return True

    @staticmethod
    def _sort_key(run) -> tuple:
        tags = run.data.tags
        selected = tags.get("selected_for_serving", "").lower() == "true"
        production_candidate = tags.get("production_candidate", "").lower() == "true"
        f1 = float(run.data.metrics.get("f1", -1.0))
        roc_auc = float(run.data.metrics.get("roc_auc", -1.0))
        return (int(selected), int(production_candidate), f1, roc_auc)

    def _find_best_run_id(self, client) -> str:
        experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            raise RuntimeError(
                f"MLflow experiment not found: {config.MLFLOW_EXPERIMENT_NAME!r}. "
                "Set MLFLOW_EXPERIMENT_NAME or MLFLOW_RUN_ID."
            )

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            max_results=100,
        )
        candidates: List[Any] = [run for run in runs if self._is_clean_finished_run(run)]
        if not candidates:
            raise RuntimeError(
                "No finished clean HW02 run with f1 and roc_auc metrics was found. "
                "Set MLFLOW_RUN_ID to the selected run manually."
            )

        best_run = sorted(candidates, key=self._sort_key, reverse=True)[0]
        return best_run.info.run_id

    def _load_model_from_run(self, client, run_id: str) -> tuple[Any, str, str]:
        import joblib
        import mlflow.sklearn

        native_model_uri = f"runs:/{run_id}/model"

        try:
            model = mlflow.sklearn.load_model(native_model_uri)
            return model, native_model_uri, "mlflow_sklearn_model"
        except Exception as native_error:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = client.download_artifacts(
                        run_id=run_id,
                        path="model_fallback/model.joblib",
                        dst_path=tmpdir,
                    )
                    model = joblib.load(model_path)
                fallback_uri = f"runs:/{run_id}/model_fallback/model.joblib"
                return model, fallback_uri, "joblib_artifact_fallback"
            except Exception as fallback_error:
                raise RuntimeError(
                    "Could not load the selected model as either a native MLflow sklearn model "
                    "or a joblib fallback artifact. "
                    f"Native error: {native_error}. Fallback error: {fallback_error}"
                ) from fallback_error

    def load(self) -> None:
        """Load the selected model once.

        This method is lock-protected because FastAPI may trigger loading from
        startup and from a prediction request at nearly the same time.
        """
        with self._load_lock:
            if self.state.loaded and self.state.model is not None:
                return

            self.state = LoadedModelState(error="Model loading in progress.")
            try:
                client = self._configure_mlflow()
                run_id = config.MLFLOW_RUN_ID or self._find_best_run_id(client)
                run = client.get_run(run_id)
                model, model_uri, load_mode = self._load_model_from_run(client, run_id)

                self.state = LoadedModelState(
                    model=model,
                    loaded=True,
                    error=None,
                    model_uri=model_uri,
                    run_id=run_id,
                    run_name=run.data.tags.get("mlflow.runName"),
                    metrics=dict(run.data.metrics),
                    params=dict(run.data.params),
                    tags={**dict(run.data.tags), "model_load_mode": load_mode},
                )
            except Exception as exc:
                self.state.loaded = False
                self.state.model = None
                self.state.error = str(exc)

    def require_model(self):
        if not self.state.loaded or self.state.model is None:
            # Retry once on demand. This keeps /docs reachable even when the
            # tracking server is temporarily slow during application startup.
            self.load()
        if not self.state.loaded or self.state.model is None:
            raise RuntimeError(self.state.error or "Model is not loaded.")
        return self.state.model

    def model_info(self) -> dict:
        return {
            "model_loaded": self.state.loaded,
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "experiment_name": config.MLFLOW_EXPERIMENT_NAME,
            "model_uri": self.state.model_uri,
            "run_id": self.state.run_id,
            "run_name": self.state.run_name,
            "dataset_version": config.DATASET_VERSION,
            "target": config.TARGET_NAME,
            "threshold": config.PREDICTION_THRESHOLD,
            "metrics": self.state.metrics,
            "params": self.state.params,
            "tags": self.state.tags,
            "error": self.state.error,
        }
