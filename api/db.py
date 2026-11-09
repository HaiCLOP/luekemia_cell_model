"""Optional Supabase persistence. If SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set, the API runs
without a database: predictions still work, nothing is logged, and /feedback returns 503.

The service-role key is read from the environment on the server only. It never reaches the web client.
"""
from __future__ import annotations

import logging
import os
import uuid

log = logging.getLogger("api.db")
BUCKET = "uploads"


class Database:
    def __init__(self):
        url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.client = None
        self.status = "disabled"
        self.model_version_id: str | None = None
        if url and key:
            try:
                from supabase import create_client

                self.client = create_client(url, key)
                self.status = "connected"
            except Exception as e:  # keep serving predictions if the DB is down
                log.error("Supabase init failed: %s", e)
                self.status = "error"

    @property
    def enabled(self) -> bool:
        return self.client is not None and self.status == "connected"

    def ensure_model_version(self, meta: dict) -> None:
        """Create the model_versions row for the served model if missing; activate it if none is active."""
        if not self.enabled:
            return
        t = self.client.table("model_versions")
        found = t.select("id,is_active").eq("name", meta["name"]).execute().data
        if found:
            self.model_version_id = found[0]["id"]
        else:
            row = {"name": meta["name"], "architecture": meta["architecture"], "git_sha": meta.get("git_sha"),
                   "checkpoint_uri": meta.get("onnx_file"), "temperature": meta["temperature"],
                   "t_low": meta["t_low"], "t_high": meta["t_high"], "band_target": meta.get("band_target"),
                   "metrics": meta.get("metrics"), "is_active": False}
            self.model_version_id = t.insert(row).execute().data[0]["id"]
        if not t.select("id").eq("is_active", True).execute().data:
            t.update({"is_active": True}).eq("id", self.model_version_id).execute()

    def log_prediction(self, sha256: str, pred, session_id: str | None) -> str:
        if not self.enabled or not self.model_version_id:
            return str(uuid.uuid4())
        row = {"model_version_id": self.model_version_id, "image_sha256": sha256, "prob_raw": pred.prob_raw,
               "prob_calibrated": pred.prob_calibrated, "decision": pred.decision, "latency_ms": pred.latency_ms,
               "session_id": session_id}
        try:
            return self.client.table("predictions").insert(row).execute().data[0]["id"]
        except Exception as e:
            log.error("prediction log failed: %s", e)
            return str(uuid.uuid4())

    def store_image(self, prediction_id: str, data: bytes, fmt: str) -> bool:
        if not self.enabled:
            return False
        ext = {"PNG": "png", "JPEG": "jpg", "BMP": "bmp"}[fmt]
        try:
            self.client.storage.from_(BUCKET).upload(f"{prediction_id}.{ext}", data,
                                                     {"content-type": f"image/{ext if ext != 'jpg' else 'jpeg'}"})
            return True
        except Exception as e:
            log.error("image upload failed: %s", e)
            return False

    def prediction_exists(self, prediction_id: str) -> bool:
        return bool(self.client.table("predictions").select("id").eq("id", prediction_id).execute().data)

    def add_feedback(self, prediction_id: str, user_label: str, comment: str | None) -> str:
        row = {"prediction_id": prediction_id, "user_label": user_label, "comment": comment}
        return self.client.table("feedback").insert(row).execute().data[0]["id"]
