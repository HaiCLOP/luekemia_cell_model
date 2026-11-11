"""Progress reporting for long jobs, shown live on the local monitor's Jobs page
(python monitor/server.py, then http://127.0.0.1:8765/jobs.html). Write the title, explanation and step
names in plain words: the page is meant to be read by someone who is not watching the code.

    with Progress("taleqani_eval", "Testing the smear reader on another hospital's smears", total=len(items),
                  explain="Checks whether the model works outside the lab it learned from.",
                  steps=["Open the photos", "Find and read every cell", "Add up the results"]) as pg:
        for item in items:
            ...
            pg.step(note=item.name, metrics={"cells": n})

Writes reports/live/jobs/<job>.json at most twice a second (and always on the last step), so it costs
nothing next to the work itself. A job that raises is marked "failed" with the error message.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

from src.utils import resolve


class Progress:
    def __init__(self, job: str, title: str, total: int, stage: str = "", explain: str = "",
                 steps: list[str] | None = None):
        """title: plain words a non-specialist understands. explain: one or two sentences on what the job does
        and why. steps: the stages in order, in plain words; set_stage(step=i) marks which one is running."""
        self.job, self.title, self.total, self.stage = job, title, max(1, int(total)), stage
        self.explain, self.steps, self.step_index = explain, steps, None
        self.done, self.metrics, self.note, self.status = 0, {}, "", "running"
        self.started = time.time()
        self.stage_started = self.started
        self._last = 0.0
        self.path = resolve("reports/live/jobs") / f"{job}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write(force=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.status, self.note = "failed", f"{exc_type.__name__}: {exc}"[:300]
        elif self.status == "running":
            self.status = "finished"
        self._write(force=True)
        return False

    def set_stage(self, stage: str, total: int | None = None, step: int | None = None) -> None:
        self.stage = stage
        if step is not None:
            self.step_index = step
        if total is not None:
            self.total, self.done = max(1, int(total)), 0
        self.stage_started = time.time()
        self._write(force=True)

    def step(self, n: int = 1, note: str = "", metrics: dict | None = None) -> None:
        self.done += n
        if note:
            self.note = note
        if metrics:
            self.metrics.update(metrics)
        self._write(force=self.done >= self.total)

    def _write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last < 0.5:
            return
        self._last = now
        elapsed = now - self.started
        in_stage = now - self.stage_started          # rate and ETA come from the current stage only
        rate = self.done / in_stage if in_stage > 0 and self.done else None
        eta = (self.total - self.done) / rate if rate and self.status == "running" else None
        body = {"job": self.job, "title": self.title, "plain_title": self.title, "explain": self.explain,
                "steps": self.steps, "step_index": self.step_index, "stage": self.stage, "status": self.status,
                "done": self.done, "total": self.total, "elapsed_s": round(elapsed, 1),
                "eta_s": None if eta is None else round(eta, 1), "rate_per_s": None if rate is None else round(rate, 3),
                "note": self.note, "metrics": self.metrics, "pid": os.getpid(),
                "started": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
                "updated": datetime.now(timezone.utc).isoformat()}
        # On Windows the replace fails while a reader (the monitor page) has the file open. Progress is a
        # convenience: retry briefly, then skip this update rather than ever stopping the real job.
        tmp = self.path.with_suffix(".tmp")
        for attempt in range(5):
            try:
                tmp.write_text(json.dumps(body, default=float))
                os.replace(tmp, self.path)
                return
            except OSError:
                time.sleep(0.05 * (attempt + 1))
