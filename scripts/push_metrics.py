"""Load training runs and validation metrics (with subject-clustered CIs) into Supabase.

Usage: python scripts/push_metrics.py
Needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (read from .env). Idempotent: a run already present
(same name in config.notes) is replaced, so re-running after new results is safe.
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from supabase import create_client

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


def ci(s: str):
    m = re.findall(r"[\d.]+", s or "")
    return (float(m[0]), float(m[1])) if len(m) == 2 else (None, None)


def main() -> None:
    load_env()
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    active = sb.table("model_versions").select("id,name").eq("is_active", True).execute().data
    mv_id = active[0]["id"] if active else None

    with (REP / "tables" / "val_comparison.csv").open(encoding="utf-8") as f:
        cmp = {r["run"]: r for r in csv.DictReader(f)}

    pushed = 0
    for p in sorted((REP / "runs").glob("*.json")):
        if p.stem.startswith("smoke"):
            continue
        d = json.loads(p.read_text())
        name = d.get("name", p.stem)
        # replace any previous copy of this run
        old = sb.table("training_runs").select("id").eq("notes", name).execute().data
        for o in old:
            sb.table("training_runs").delete().eq("id", o["id"]).execute()
        cfg = d.get("config", {"note": "baseline A has no training config"})
        seed = d.get("seed", 42)
        is_member = name.startswith("effb0_main_s")
        run = sb.table("training_runs").insert({
            "model_version_id": mv_id if is_member else None, "seed": seed, "config": cfg, "status": "finished",
            "notes": name, "started_at": d.get("started"), "finished_at": d.get("finished"),
        }).execute().data[0]
        rows = []
        c = cmp.get(name)
        if c:
            for m in ("pr_auc", "roc_auc", "recall", "specificity", "ece", "brier"):
                lo, hi = ci(c.get(f"{m}_ci"))
                rows.append({"run_id": run["id"], "split": "val", "metric": m, "value": float(c[m]), "ci_low": lo, "ci_high": hi})
            rows.append({"run_id": run["id"], "split": "val", "metric": "subject_acc", "value": float(c["subject_acc"])})
        else:
            for m, v in (d.get("val_metrics") or {}).items():
                if isinstance(v, (int, float)):
                    rows.append({"run_id": run["id"], "split": "val", "metric": m, "value": float(v)})
        if rows:
            sb.table("run_metrics").insert(rows).execute()
        pushed += 1
        print(f"pushed {name}: {len(rows)} metrics")
    print(f"done: {pushed} runs")


if __name__ == "__main__":
    main()
