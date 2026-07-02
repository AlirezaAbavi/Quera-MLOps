"""app.model_loader — wraps the HW3_A bundle's predict.py.

The bundle is the unit of truth. This module:
  1. Discovers the bundle on disk (env BUNDLE_DIR or ../HW3_A/bundle).
  2. Imports the bundle predictor.
  3. Verifies the MANIFEST.json SHAs at startup.
  4. Holds one global predictor instance.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_DEFAULT_BUNDLE_IN_IMAGE = "/app/bundle"
_DEV_BUNDLE = str(
    Path(__file__).resolve().parent.parent.parent / "HW3_A" / "bundle"
)


def _resolve_bundle_dir() -> Path:
    """Resolve bundle directory using env, container path, then local dev path."""
    env = os.getenv("BUNDLE_DIR", "").strip()

    candidates = []

    if env:
        candidates.append(Path(env).expanduser().resolve())

    candidates.append(Path(_DEFAULT_BUNDLE_IN_IMAGE).expanduser().resolve())
    candidates.append(Path(_DEV_BUNDLE).expanduser().resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    checked = "\n".join(str(path) for path in candidates)

    raise FileNotFoundError(
        "Could not find encoder bundle directory. Checked:\n"
        f"{checked}\n\n"
        "Set BUNDLE_DIR to the HW3_A bundle path, for example:\n"
        "BUNDLE_DIR=../HW3_A/bundle"
    )


def _sha256(path: Path) -> str:
    """Compute SHA-256 hash for a file using 1MB chunks."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def _verify_manifest(bundle_dir: Path) -> tuple[bool, str]:
    """Verify MANIFEST.json entries exist and match recorded SHA-256 hashes."""
    manifest_path = bundle_dir / "MANIFEST.json"

    if not manifest_path.exists():
        return False, f"MANIFEST.json not found at {manifest_path}"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Could not parse MANIFEST.json: {exc}"

    files = manifest.get("files", {})

    if not isinstance(files, dict) or not files:
        return False, "MANIFEST.json has no valid files mapping"

    checked = 0

    for rel, expected in files.items():
        if not isinstance(expected, str):
            return False, f"Invalid hash for {rel}: expected string"

        if expected.startswith("REPLACE"):
            return False, f"Placeholder hash found for {rel}"

        path = bundle_dir / rel

        if not path.exists():
            return False, f"Manifest file missing on disk: {rel}"

        actual = _sha256(path)

        if actual != expected:
            return False, (
                f"SHA mismatch for {rel}: "
                f"expected {expected}, got {actual}"
            )

        checked += 1

    return True, f"{checked} files OK"


class _FunctionBundlePredictor:
    """Compatibility wrapper for HW3_A bundles that expose functions, not a class.

    Some bundle implementations expose:
      load_bundle(), embed(), similarity(), info()

    HW3_B uses BUNDLE_DIR as the bundle root:
      ../HW3_A/bundle

    HW3_A predict.py uses BUNDLE_DIR as the model directory:
      ../HW3_A/bundle/model

    This wrapper bridges that difference.
    """

    def __init__(self, module, bundle_dir: Path):
        self.module = module
        self.bundle_dir = bundle_dir
        self.model_dir = bundle_dir / "model"

        # The HW3_A predict.py reads env BUNDLE_DIR inside embed().
        # So we force it to point to bundle/model, not bundle/.
        os.environ["BUNDLE_DIR"] = str(self.model_dir)

        if hasattr(module, "load_bundle"):
            module.load_bundle(str(self.model_dir))

    def embed(self, texts):
        # Keep env correct even if another part of the app changed it.
        os.environ["BUNDLE_DIR"] = str(self.model_dir)
        return self.module.embed(list(texts))

    def info(self):
        os.environ["BUNDLE_DIR"] = str(self.model_dir)

        if hasattr(self.module, "info"):
            return self.module.info()

        return {}


@dataclass
class LoadState:
    loaded: bool = False
    error: Optional[str] = None
    bundle_dir: Optional[Path] = None
    manifest_ok: Optional[bool] = None
    manifest_msg: Optional[str] = None


@dataclass
class ModelService:
    state: LoadState = field(default_factory=LoadState)
    predictor: Optional[object] = None
    metadata: dict = field(default_factory=dict)

    def load(self) -> None:
        """Load and verify the HW3_A encoder bundle."""
        self.state = LoadState()
        self.predictor = None
        self.metadata = {}

        try:
            bundle_dir = _resolve_bundle_dir()
            self.state.bundle_dir = bundle_dir

            manifest_ok, manifest_msg = _verify_manifest(bundle_dir)
            self.state.manifest_ok = manifest_ok
            self.state.manifest_msg = manifest_msg

            if not manifest_ok:
                raise RuntimeError(f"Bundle manifest verification failed: {manifest_msg}")

            model_dir = bundle_dir / "model"

            if not model_dir.exists() or not model_dir.is_dir():
                raise FileNotFoundError(f"Bundle model directory not found: {model_dir}")

            required_model_files = [
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.txt",
                "special_tokens_map.json",
                "model.safetensors",
            ]

            missing_model_files = [
                fname for fname in required_model_files
                if not (model_dir / fname).exists()
            ]

            if missing_model_files:
                raise FileNotFoundError(
                    "Bundle model directory is incomplete. Missing: "
                    + ", ".join(missing_model_files)
                )

            predict_path = bundle_dir / "predict.py"

            if not predict_path.exists():
                raise FileNotFoundError(f"Bundle predict.py not found: {predict_path}")

            bundle_dir_str = str(bundle_dir)

            if bundle_dir_str not in sys.path:
                sys.path.insert(0, bundle_dir_str)

            # Make sure we import the current bundle/predict.py, not a stale module.
            if "predict" in sys.modules:
                del sys.modules["predict"]

            predict_module = importlib.import_module("predict")

            if hasattr(predict_module, "BundlePredictor"):
                predictor_cls = getattr(predict_module, "BundlePredictor")
                self.predictor = predictor_cls(bundle_dir=model_dir)
            else:
                self.predictor = _FunctionBundlePredictor(
                    module=predict_module,
                    bundle_dir=bundle_dir,
                )

            meta_path = bundle_dir / "metadata.json"

            if meta_path.exists():
                self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                self.metadata = {}

            self.state.loaded = True
            self.state.error = None

        except Exception as exc:
            self.state.loaded = False
            self.state.error = str(exc)
            self.predictor = None

    def require_predictor(self) -> object:
        """Return loaded predictor or raise if unavailable."""
        if not self.state.loaded or self.predictor is None:
            raise RuntimeError(self.state.error or "model not loaded")

        return self.predictor

    def info(self) -> dict:
        """Return bundle status and metadata for /model-info."""
        return {
            "bundle_loaded": self.state.loaded,
            "bundle_dir": str(self.state.bundle_dir) if self.state.bundle_dir else None,
            "manifest_ok": self.state.manifest_ok,
            "manifest_msg": self.state.manifest_msg,
            "error": self.state.error,
            "metadata": self.metadata,
        }