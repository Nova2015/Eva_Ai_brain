"""Bounded local training worker for approved EVA datasets.

The first executable adapter is tabular XGBoost. Other modalities are rejected
explicitly until their dedicated adapters and evaluation contracts exist.
"""
from __future__ import annotations

import json
import math
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .processor import sha256_file
from .registry import registry
from .config import get_dataset_root
from .path_security import validate_path


class _WorkerHeartbeat:
    """Keep a claimed job lease alive for the duration of worker execution."""

    def __init__(self, job_id: str, worker_token: str, registry_instance: Any):
        self.job_id = job_id
        self.worker_token = worker_token
        self.registry = registry_instance
        try:
            configured_interval = float(
                os.getenv("EVA_DL_WORKER_HEARTBEAT_INTERVAL_SECONDS", "10")
            )
        except (TypeError, ValueError):
            configured_interval = 10.0
        if not math.isfinite(configured_interval) or configured_interval <= 0:
            configured_interval = 10.0
        self.interval = max(1.0, configured_interval)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"eva-dl-heartbeat-{job_id[-8:]}",
            daemon=True,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                still_owner = self.registry.heartbeat_job(
                    self.job_id, worker_token=self.worker_token
                )
            except Exception:
                # Transient SQLite errors must not falsely declare a process dead.
                continue
            if not still_owner:
                return

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval + 1.0)



def _tabular_rows(prepared: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    tables = prepared.get("tables") or []
    if not tables:
        raise ValueError("No tabular data was extracted from this dataset")
    # CSV/JSON/DOCX tables are represented as row arrays. XLSX stores one
    # sheet container per table; the first sheet is selected explicitly.
    table = tables[0]
    if len(table) == 1 and isinstance(table[0], dict) and "rows" in table[0]:
        table = table[0]["rows"]
    if not isinstance(table, list) or len(table) < 3:
        raise ValueError("At least two data rows plus a header are required")
    headers = [str(value).strip() for value in table[0]]
    if not headers or len(headers) != len(set(headers)):
        raise ValueError("Column headers must be non-empty and unique")
    records = []
    for row in table[1:]:
        values = list(row)[:len(headers)] + [None] * max(0, len(headers)-len(row))
        records.append(dict(zip(headers, values)))
    return headers, records


def _train_tabular(prepared: dict[str, Any], review: dict[str, Any], device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("XGBoost is not installed; tabular training adapter unavailable") from exc

    headers, records = _tabular_rows(prepared)
    target = review.get("target_column")
    if target not in headers:
        raise ValueError("Reviewed target column is missing from the prepared dataset")
    features = review.get("feature_columns") or [name for name in headers if name != target]
    if not features or target in features or any(name not in headers for name in features):
        raise ValueError("Reviewed feature columns are invalid")
    task = review.get("task_type")
    if task not in {"classification", "regression"}:
        raise ValueError("The tabular worker supports classification and regression only")

    cleaned = []
    for record in records:
        label = record.get(target)
        if label is None or str(label).strip() == "":
            continue
        cleaned.append(record)
    if len(cleaned) < 10:
        raise ValueError("At least 10 labelled rows are required for a train/validation split")

    encoders: dict[str, dict[str, int]] = {}
    medians: dict[str, float] = {}
    feature_kinds: dict[str, str] = {}
    for name in features:
        values = [record.get(name) for record in cleaned if record.get(name) not in (None, "")]
        numeric_values = []
        try:
            numeric_values = [float(value) for value in values]
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError
            feature_kinds[name] = "numeric"
        except (TypeError, ValueError):
            feature_kinds[name] = "categorical"
            encoders[name] = {key: index for index, key in enumerate(sorted({str(value) for value in values}))}
    matrix = []
    for record in cleaned:
        row = []
        for name in features:
            value = record.get(name)
            if value is None or str(value).strip() == "":
                row.append(float("nan"))
            elif feature_kinds[name] == "numeric":
                row.append(float(value))
            else:
                row.append(float(encoders[name][str(value)]))
        matrix.append(row)
    x = np.asarray(matrix, dtype=np.float32)
    for index, name in enumerate(features):
        column = x[:, index]
        finite = column[np.isfinite(column)]
        median = float(np.median(finite)) if finite.size else 0.0
        medians[name] = median
        column[~np.isfinite(column)] = median
    raw_y = [record[target] for record in cleaned]
    label_mapping = None
    if task == "classification":
        labels = sorted({str(value) for value in raw_y})
        if len(labels) < 2 or len(labels) > 100:
            raise ValueError("Classification requires between 2 and 100 distinct target classes")
        label_mapping = {label: index for index, label in enumerate(labels)}
        y = np.asarray([label_mapping[str(value)] for value in raw_y], dtype=np.int32)
    else:
        try:
            y = np.asarray([float(value) for value in raw_y], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Regression target must contain numeric values") from exc
        if not np.isfinite(y).all():
            raise ValueError("Regression target contains non-finite values")

    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    test_count = max(2, int(round(len(y) * 0.2)))
    train_indices, test_indices = order[test_count:], order[:test_count]
    if task == "classification" and len(set(y[train_indices].tolist())) < 2:
        raise ValueError("Training split does not contain at least two classes")
    if device == "cuda":
        # The worker never silently moves a requested GPU run to CPU.
        import torch
        import xgboost
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA became unavailable before training began")
        raw_cuda_flag = xgboost.build_info().get("USE_CUDA", False)
        if not (raw_cuda_flag is True or str(raw_cuda_flag).strip().lower() in {"1", "true", "on", "yes"}):
            raise RuntimeError("Installed XGBoost build does not support CUDA")
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        minimum_vram = int(os.getenv("EVA_DL_MIN_FREE_VRAM_MB", "1024"))
        if free_bytes < minimum_vram * 1024 * 1024:
            raise RuntimeError("Free GPU memory fell below the training safety threshold")
        model_device = "cuda"
    else:
        model_device = "cpu"

    train_matrix = xgb.DMatrix(x[train_indices], label=y[train_indices])
    validation_matrix = xgb.DMatrix(x[test_indices], label=y[test_indices])
    common = {"max_depth": 4, "eta": 0.05, "subsample": 0.9,
              "colsample_bytree": 0.9, "nthread": 1, "seed": 42,
              "tree_method": "hist", "device": model_device}
    if task == "classification":
        if len(label_mapping) == 2:
            common.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        else:
            common.update({"objective": "multi:softprob", "num_class": len(label_mapping),
                           "eval_metric": "mlogloss"})
    else:
        common.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

    # Use XGBoost's native Booster API rather than its scikit-learn wrappers.
    # This avoids estimator-tag incompatibilities across sklearn releases.
    model = xgb.train(common, train_matrix, num_boost_round=120)

    # Capture execution evidence from the live Booster before serialization.
    # XGBoost model files may reload with a CPU default even when training ran
    # on CUDA, so artifact reload configuration is not a reliable GPU audit.
    training_config = json.loads(model.save_config())

    def _config_values(node: Any, key: str) -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            for child_key, child_value in node.items():
                if child_key == key and child_value is not None:
                    found.append(str(child_value))
                found.extend(_config_values(child_value, key))
        elif isinstance(node, list):
            for child_value in node:
                found.extend(_config_values(child_value, key))
        return found

    device_values = _config_values(training_config, "device")
    updater_values = _config_values(training_config, "updater")
    actual_device = next((value for value in device_values if "cuda" in value.lower()),
                         device_values[-1] if device_values else "unknown")
    actual_updater = next((value for value in updater_values if "gpu" in value.lower()),
                          updater_values[-1] if updater_values else "unknown")
    execution = {
        "requested_device": device,
        "actual_device": actual_device,
        "actual_updater": actual_updater,
        "xgboost_version": xgb.__version__,
        "xgboost_cuda_build": bool(
            xgb.build_info().get("USE_CUDA", False) is True
            or str(xgb.build_info().get("USE_CUDA", False)).strip().lower()
            in {"1", "true", "on", "yes"}
        ),
    }
    if device == "cuda":
        if not any("cuda" in value.lower() for value in device_values):
            raise RuntimeError(
                "CUDA was requested but the trained Booster reports no CUDA device"
            )
        if not any("grow_gpu_hist" in value.lower() for value in updater_values):
            raise RuntimeError(
                "CUDA was requested but the trained Booster reports no GPU histogram updater"
            )
        import torch
        execution["gpu_name"] = torch.cuda.get_device_name(0)

    predictions = model.predict(validation_matrix)
    if task == "classification":
        if len(label_mapping) == 2:
            predicted_labels = (predictions >= 0.5).astype(np.int32)
        else:
            predicted_labels = np.asarray(predictions).reshape(-1, len(label_mapping)).argmax(axis=1)
        metrics = {"accuracy": float(np.mean(predicted_labels == y[test_indices]))}
    else:
        rmse = float(np.sqrt(np.mean((predictions.astype(float)-y[test_indices]) ** 2)))
        metrics = {"rmse": rmse, "mae": float(np.mean(np.abs(predictions.astype(float)-y[test_indices])))}
    metrics.update({"train_rows": int(len(train_indices)), "validation_rows": int(len(test_indices))})
    preprocessing = {"feature_columns": features, "target_column": target,
                     "missing_value_medians": medians, "categorical_encoders": encoders, "feature_kinds": feature_kinds,
                     "label_mapping": label_mapping, "split_seed": 42, "validation_fraction": 0.2}
    return model, {"metrics": metrics, "preprocessing": preprocessing,
                   "model_family": "XGBoost", "execution": execution}



def run_training_job(job_id: str, *, owner_id: str, workspace_id: str,
                    already_claimed: bool = False) -> dict[str, Any]:
    """Execute one approved local job. Errors are persisted without leaking paths."""
    job = registry.get_job(job_id, owner_id=owner_id, workspace_id=workspace_id)
    if job is None:
        raise KeyError("Training job not found")

    if already_claimed:
        if job["status"] != "running":
            return job
    else:
        if job["status"] != "queued":
            return job
        if not registry.claim_job(job_id, owner_id=owner_id, workspace_id=workspace_id):
            return registry.get_job(job_id, owner_id=owner_id, workspace_id=workspace_id) or job

    heartbeat = None
    artifact_path: Path | None = None
    model_registered = False

    try:
        lease_getter = getattr(registry, "get_worker_lease", None)
        lease = lease_getter(job_id) if callable(lease_getter) else None
        worker_token = (lease or {}).get("worker_token")
        heartbeat = (
            _WorkerHeartbeat(job_id, worker_token, registry)
            if worker_token and callable(getattr(registry, "heartbeat_job", None))
            else None
        )
        if heartbeat:
            heartbeat.start()
        dataset = registry.get_dataset(
            job["dataset_id"], owner_id=owner_id, workspace_id=workspace_id
        )
        if dataset is None:
            raise KeyError("Training dataset not found")

        from src.data_governance import get_record

        governed = get_record(dataset["data_id"])
        if (
            not governed
            or governed.get("owner_id") != owner_id
            or governed.get("workspace_id") != workspace_id
        ):
            raise PermissionError("Governed dataset scope verification failed")

        if governed.get("training_approval") != "approved":
            raise PermissionError("Explicit Data Governance training approval is required")

        if (
            governed.get("sha256") != dataset["sha256"]
            or int(governed.get("version", 0)) != int(dataset["source_version"])
        ):
            raise ValueError("Dataset source version changed; re-prepare and review it")

        source = validate_path(dataset["file_path"], must_exist=True, regular_file=True)
        if sha256_file(source) != dataset["sha256"]:
            raise ValueError("Dataset source file is missing or its integrity hash changed")

        metadata = json.loads(dataset.get("metadata_json") or "{}")
        review = metadata.get("label_review") or {}

        if review.get("status") != "verified":
            raise ValueError("Dataset labels and target columns have not been reviewed")

        dataset_root = validate_path(get_dataset_root(), directory=True)
        processed_dir = validate_path(dataset_root / "processed", directory=True)
        artifact_dir = validate_path(processed_dir / dataset["dataset_id"], directory=True)
        prepared_path = validate_path(
            artifact_dir / "prepared.json", must_exist=True, regular_file=True
        )
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        if prepared.get("source_sha256") != dataset["sha256"]:
            raise ValueError("Prepared artifact does not match the approved source hash")

        registry.update_job(
            job_id, owner_id=owner_id, workspace_id=workspace_id,
            status="running", stage="validating", progress=5,
        )

        if review.get("modality") != "tabular":
            raise NotImplementedError(
                "This worker release executes tabular datasets only; "
                "other modality adapters are not yet enabled"
            )

        registry.update_job(
            job_id, owner_id=owner_id, workspace_id=workspace_id,
            status="running", stage="training", progress=20,
        )

        model, result = _train_tabular(prepared, review, job["selected_device"])
        model_id = f"EVA-MODEL-{uuid.uuid4().hex.upper()}"
        version = 1

        artifact_dir = validate_path(artifact_dir, directory=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = validate_path(artifact_dir, must_exist=True, directory=True)
        artifact_path = validate_path(artifact_dir / f"{model_id}_v{version}.json")
        registry.record_artifact_intent(
            job_id,
            owner_id=owner_id,
            workspace_id=workspace_id,
            dataset_id=dataset["dataset_id"],
            model_id=model_id,
            version=version,
            artifact_path=str(artifact_path),
        )
        validate_path(artifact_dir, must_exist=True, directory=True)
        artifact_path = validate_path(artifact_path)
        model.save_model(str(artifact_path))

        registry.update_job(
            job_id, owner_id=owner_id, workspace_id=workspace_id,
            status="running", stage="evaluating", progress=85,
        )

        registry.create_model(
            model_id=model_id,
            version=version,
            dataset_id=dataset["dataset_id"],
            data_id=dataset["data_id"],
            owner_id=owner_id,
            workspace_id=workspace_id,
            model_family=result["model_family"],
            task_type=review["task_type"],
            artifact_path=str(artifact_path),
            metrics=result["metrics"],
            preprocessing=result["preprocessing"],
            status="candidate",
        )
        model_registered = True

        return registry.update_job(
            job_id, owner_id=owner_id, workspace_id=workspace_id,
            status="completed", stage="completed", progress=100,
            metadata={
                "model_id": model_id,
                "model_version": version,
                "metrics": result["metrics"],
                "execution": result["execution"],
            },
        )

    except Exception as exc:
        failure_message = str(exc)[:450]
        failure_metadata = {"failure_message": failure_message}

        # Persist failure first. Cleanup is then coordinated by the registry's
        # atomic claim, never by a direct worker-side unlink().
        if artifact_path is not None:
            failure_metadata["artifact_cleanup"] = {
                "status": "pending",
                "created_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            }

        registry.update_job(
            job_id, owner_id=owner_id, workspace_id=workspace_id,
            status="failed", stage="failed", progress=0,
            error_code=type(exc).__name__,
            metadata=failure_metadata,
        )

        if artifact_path is not None:
            cleanup = getattr(registry, "cleanup_failed_job_artifact", None)
            if callable(cleanup):
                try:
                    cleanup(job_id, require_verified_recovery=False)
                except Exception:
                    # Preserve and re-raise the original training exception.
                    pass
        raise
    finally:
        if heartbeat:
            heartbeat.stop()

def _execute_job_safely(job_id: str, *, owner_id: str, workspace_id: str,
                        already_claimed: bool = False) -> None:
    try:
        run_training_job(job_id, owner_id=owner_id, workspace_id=workspace_id,
                         already_claimed=already_claimed)
    except Exception as exc:
        current = registry.get_job(job_id, owner_id=owner_id, workspace_id=workspace_id)
        if current and current.get("status") == "running":
            registry.update_job(
                job_id, owner_id=owner_id, workspace_id=workspace_id,
                status="failed", stage="failed", progress=0,
                error_code=type(exc).__name__,
                metadata={"failure_message": str(exc)[:500]},
            )


def run_training_job_safely(job_id: str, *, owner_id: str, workspace_id: str) -> None:
    """Recover verified dead workers, run one job, then drain the local queue."""
    recovery = getattr(registry, "recover_stale_jobs", None)
    if callable(recovery):
        recovery()
    _execute_job_safely(job_id, owner_id=owner_id, workspace_id=workspace_id)
    while True:
        queued = registry.get_next_queued_job()
        if queued is None:
            return
        if not registry.claim_job(queued["job_id"], owner_id=queued["owner_id"],
                                  workspace_id=queued["workspace_id"]):
            return
        _execute_job_safely(queued["job_id"], owner_id=queued["owner_id"],
                            workspace_id=queued["workspace_id"], already_claimed=True)