"""Local training monitor. Standard library only.

Usage:  .venv\\Scripts\\python.exe monitor\\server.py   (then open http://127.0.0.1:8765)
Binds to 127.0.0.1 only. Read-only: it never writes to the project.
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
STATIC = Path(__file__).resolve().parent / "static"
PORT = int(os.environ.get("PORT", "8765"))
QUEUE_SCRIPTS = [ROOT / "scripts" / "queue_experiments.ps1"]

EPOCH_RE = re.compile(
    r"\[(?P<stage>\w+) ep(?P<epoch>\d+)\] loss (?P<loss>[\d.]+)\s+val pr_auc (?P<pr_auc>[\d.]+|nan)\s+"
    r"roc_auc (?P<roc_auc>[\d.]+|nan)\s+recall (?P<recall>[\d.]+)\s+ece (?P<ece>[\d.]+)\s+(?P<sec>\d+)s"
)
QUEUE_START = re.compile(r"=== (?P<name>\S+) started (?P<when>.+)$")
QUEUE_END = re.compile(r"=== (?P<name>\S+) exit (?P<code>-?\d+) after (?P<min>[\d.]+) min")
QUEUE_SKIP = re.compile(r"=== (?P<name>\S+) skipped")

# Rough per-run minutes on a GTX 1650, used until a run of the same family has finished.
DEFAULT_MIN = [("baseline_a", 4), ("baseline_b", 33), ("abl_frozen", 7), ("dinov2", 70), ("mnv3", 22),
               ("effb0", 24), ("abl_", 24)]
FAMILY = [("baseline_a", "Frozen features + LR"), ("baseline_b", "Small CNN (scratch)"),
          ("effb0_main", "EfficientNet-B0"), ("mnv3", "MobileNetV3-L"), ("dinov2", "DINOv2 ViT-S/14"),
          ("abl_frozen", "Ablation: frozen backbone"), ("abl_aug_none", "Ablation: no augmentation"),
          ("abl_aug_hed", "Ablation: + HED stain aug"), ("abl_sampler", "Ablation: weighted sampler"),
          ("abl_labelsmooth", "Ablation: label smoothing 0.1"), ("smoke", "Smoke test")]
STAGE_EPOCHS = {"stage1": 5, "stage2": 25, "scratch": 30}


def read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if b"\x00" in raw:  # UTF-16 LE from PowerShell, possibly with UTF-8 appended; content is ASCII
        raw = raw.replace(b"\xff\xfe", b"").replace(b"\xfe\xff", b"").replace(b"\x00", b"")
    return raw.replace(b"\xef\xbb\xbf", b"").decode("utf-8", errors="replace")


def num(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def label_for(name: str) -> str:
    for prefix, lab in FAMILY:
        if name.startswith(prefix):
            seed = re.search(r"_s(\d+)$", name)
            return lab + (f" · seed {seed.group(1)}" if seed and prefix == "effb0_main" else "")
    return name


def family_of(name: str) -> str:
    return re.sub(r"_s\d+$", "", name)


def parse_when(s: str | None):
    if not s:
        return None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %I:%M:%S %p"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def parse_queue() -> list[dict]:
    jobs: dict[str, dict] = {}
    order: list[str] = []
    for line in read_text(REPORTS / "logs" / "queue.log").splitlines():
        line = line.strip()
        if m := QUEUE_START.match(line):
            n = m["name"]
            if n not in order:
                order.append(n)
            jobs[n] = {"name": n, "status": "running", "started": m["when"].strip(), "minutes": None, "exit": None}
        elif m := QUEUE_END.match(line):
            n = m["name"]
            if n not in order:
                order.append(n)
            j = jobs.setdefault(n, {"name": n, "started": None})
            j.update(status="done" if m["code"] == "0" else "failed", exit=int(m["code"]), minutes=float(m["min"]))
        elif m := QUEUE_SKIP.match(line):
            n = m["name"]
            if n not in jobs:
                order.append(n)
                jobs[n] = {"name": n, "status": "done", "started": None, "minutes": None, "exit": 0}
    return [jobs[n] for n in order]


def planned_names() -> list[str]:
    """Run names in the order the queue scripts will execute them."""
    names: list[str] = []
    for script in QUEUE_SCRIPTS:
        for line in read_text(script).splitlines():
            if line.strip().startswith("#"):
                continue
            if m := re.search(r"foreach \(\$s in ([\d,\s]+)\)\s*\{\s*Run \"([\w$]+)\"", line):
                for s in re.findall(r"\d+", m.group(1)):
                    names.append(m.group(2).replace("$s", s))
            elif m := re.search(r"^\s*Run \"([\w]+)\"", line):
                names.append(m.group(1))
    return names


def parse_epochs(name: str) -> list[dict]:
    rows = []
    for line in read_text(REPORTS / "logs" / f"{name}.log").splitlines():
        if m := EPOCH_RE.search(line):
            rows.append({"stage": m["stage"], "epoch": int(m["epoch"]), "train_loss": num(m["loss"]),
                         "val_pr_auc": num(m["pr_auc"]), "val_roc_auc": num(m["roc_auc"]),
                         "val_recall": num(m["recall"]), "val_ece": num(m["ece"]), "sec": int(m["sec"])})
    return rows


def log_tail(name: str, n: int = 14) -> list[str]:
    keep = [l.rstrip() for l in read_text(REPORTS / "logs" / f"{name}.log").splitlines()
            if l.strip() and not re.search(r"warn|unauthenticated|faster downloads|RemoteException|NativeCommandError|"
                                            r"CategoryInfo|FullyQualifiedErrorId|^\s*\+|At .*\.ps1", l, re.I)]
    return keep[-n:]


def live(name: str) -> dict | None:
    try:
        d = json.loads((REPORTS / "live" / f"{name}.json").read_text())
        d["age_sec"] = round(time.time() - d.get("updated", 0), 1)
        return d
    except (OSError, ValueError):
        return None


def comparison() -> dict[str, dict]:
    p = REPORTS / "tables" / "val_comparison.csv"
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8") as f:
        return {r["run"]: r for r in csv.DictReader(f)}


def ci(s: str | None):
    if not s:
        return None
    m = re.findall(r"[\d.]+", s)
    return [float(m[0]), float(m[1])] if len(m) == 2 else None


def runs(cmp: dict) -> list[dict]:
    out = []
    for p in sorted((REPORTS / "runs").glob("*.json")):
        if p.stem.startswith("smoke"):
            continue
        try:
            d = json.loads(p.read_text())
        except ValueError:
            continue
        vm = d.get("val_metrics", {})
        c = cmp.get(d.get("name", p.stem), {})
        log = d.get("log", [])
        best_ep = None
        if log:
            best_i = max(range(len(log)), key=lambda i: log[i].get("val_pr_auc") or -1)
            best_ep = best_i + 1
        out.append({
            "name": d.get("name", p.stem), "label": label_for(d.get("name", p.stem)), "arch": d.get("arch"),
            "finished": d.get("finished"), "epochs": len(log), "best_epoch": best_ep, "vram_mb": d.get("max_vram_mb"),
            "seed": d.get("seed"),
            **{k: vm.get(k) for k in ("pr_auc", "roc_auc", "recall", "specificity", "ece", "f1", "brier")},
            "subject_acc": num(c.get("subject_acc")),
            "ci": {k: ci(c.get(f"{k}_ci")) for k in ("pr_auc", "roc_auc", "recall", "specificity", "ece")},
            "curve": [r.get("val_pr_auc") for r in log],
        })
    return out


_DATA_CACHE: dict = {}


def dataset() -> dict | None:
    p = ROOT / "data" / "splits.csv"
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    if _DATA_CACHE.get("mtime") == mtime:
        return _DATA_CACHE["data"]
    splits: dict[str, dict] = {}
    subjects, pids = set(), set()
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = splits.setdefault(r["split"], {"images": 0, "all": 0, "hem": 0, "subjects": set()})
            s["images"] += 1
            s["all" if r["label"] == "1" else "hem"] += 1
            s["subjects"].add(r.get("group_id") or r["patient_id"])
            subjects.add(r.get("group_id") or r["patient_id"])
            pids.add(r["patient_id"])
    audit = {}
    for key, fname in (("corrupt", "corrupt_dropped.csv"), ("duplicates", "duplicates_dropped.csv")):
        t = REPORTS / "tables" / fname
        audit[key] = max(0, len(read_text(t).splitlines()) - 1) if t.exists() else None
    g = REPORTS / "tables" / "patient_groups.csv"
    merged = []
    if g.exists():
        with g.open(newline="", encoding="utf-8") as f:
            merged = [r["patients"] for r in csv.DictReader(f) if int(r["n_patients"]) > 1]
    data = {
        "images": sum(s["images"] for s in splits.values()), "subjects": len(subjects), "patient_ids": len(pids),
        "splits": {k: {"images": v["images"], "all": v["all"], "hem": v["hem"], "subjects": len(v["subjects"])}
                   for k, v in sorted(splits.items(), key=lambda kv: ["train", "val", "test"].index(kv[0]))},
        "audit": {**audit, "merged_groups": merged},
    }
    _DATA_CACHE.update(mtime=mtime, data=data)
    return data


def est_minutes(name: str, history: dict[str, list[float]]) -> float:
    fam = family_of(name)
    if history.get(fam):
        return sum(history[fam]) / len(history[fam])
    for prefix, m in DEFAULT_MIN:
        if name.startswith(prefix):
            return m
    return 25


def current_detail(name: str, epochs: list[dict], lv: dict | None) -> dict:
    """ETA and early-stopping state for the running job."""
    d: dict = {"patience_limit": 8 if name.startswith("baseline_b") else 5}
    if epochs:
        stage = epochs[-1]["stage"]
        in_stage = [e for e in epochs if e["stage"] == stage]
        best_i = max(range(len(in_stage)), key=lambda i: in_stage[i]["val_pr_auc"] or -1)
        d["since_best"] = len(in_stage) - 1 - best_i
        d["stage"] = stage
        d["stage_epochs_done"] = len(in_stage)
        recent = [e["sec"] for e in in_stage[-3:]]
        d["sec_per_epoch"] = sum(recent) / len(recent)
    if lv:
        d["stage"] = lv.get("stage", d.get("stage"))
        d["stage_epoch"] = lv.get("epoch", 0) + 1
        d["stage_total"] = lv.get("epochs")
        d["batch"], d["batches"] = lv.get("batch"), lv.get("batches")
        d["img_per_sec"] = lv.get("img_per_sec")
        d["live_loss"] = lv.get("train_loss")
        d["lr"] = lv.get("lr")
        d["age_sec"] = lv.get("age_sec")
    # worst-case remaining time = remaining epochs in this stage (+ stage 2 if still in stage 1)
    spe = d.get("sec_per_epoch") or 60
    st = d.get("stage")
    done = d.get("stage_epochs_done", 0)
    if st in STAGE_EPOCHS:
        rem_epochs = STAGE_EPOCHS[st] - done + (STAGE_EPOCHS["stage2"] if st == "stage1" else 0)
        if lv and lv.get("batches"):
            rem_epochs -= lv["batch"] / lv["batches"]
        d["max_remaining_min"] = max(0.0, rem_epochs * spe / 60)
        # likely: early stopping typically ends stage 2 around epoch 15-20
        likely = rem_epochs if st != "stage2" else min(rem_epochs, max(0, 18 - done))
        if st == "stage1":
            likely = STAGE_EPOCHS["stage1"] - done + 16
        d["likely_remaining_min"] = max(0.0, likely * spe / 60)
    return d


def gpu() -> dict | None:
    q = "name,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.max.sm"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        vals = [v.strip() for v in r.stdout.strip().split(",")]
        keys = ["name", "util", "temp", "mem_used", "mem_total", "power", "power_limit", "clock", "clock_max"]
        d = dict(zip(keys, vals))
        for k in keys[1:]:
            d[k] = num(d.get(k))
        return d
    except Exception:
        return None


def jobs() -> list[dict]:
    """Long jobs that report through src/progress.py. A 'running' job whose process is gone is shown as stale."""
    out = []
    for f in sorted((REPORTS / "live" / "jobs").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if d.get("status") == "running":
            try:
                age = time.time() - datetime.fromisoformat(d["updated"]).timestamp()
            except Exception:
                age = 0
            if age > 120:
                d["status"] = "stale"
        out.append(d)
    order = {"running": 0, "stale": 1, "failed": 2, "finished": 3}
    return sorted(out, key=lambda d: (order.get(d.get("status"), 9), d.get("updated", "")), reverse=False)


def status() -> dict:
    queue = parse_queue()
    by_name = {j["name"]: j for j in queue}
    current = next((j["name"] for j in reversed(queue) if j["status"] == "running"), None)
    focus = current or (queue[-1]["name"] if queue else None)
    cmp = comparison()
    run_list = runs(cmp)
    run_by_name = {r["name"]: r for r in run_list}

    history: dict[str, list[float]] = {}
    for j in queue:
        if j["status"] == "done" and j.get("minutes"):
            history.setdefault(family_of(j["name"]), []).append(j["minutes"])

    epochs = parse_epochs(focus) if focus else []
    lv = live(focus) if focus else None
    detail = current_detail(focus, epochs, lv) if focus else {}

    # timeline = everything already in queue.log, then planned jobs not yet started
    names = [j["name"] for j in queue] + [n for n in planned_names() if n not in by_name]
    now = datetime.now()
    timeline, cursor = [], None
    for n in names:
        j = by_name.get(n, {"name": n, "status": "pending"})
        item = {"name": n, "label": label_for(n), "status": j["status"], "minutes": j.get("minutes"),
                "pr_auc": (run_by_name.get(n) or {}).get("pr_auc")}
        if j["status"] == "running":
            started = parse_when(j.get("started"))
            elapsed = (now - started).total_seconds() / 60 if started else 0
            rem = detail.get("likely_remaining_min", est_minutes(n, history) - elapsed)
            item.update(elapsed=elapsed, est=elapsed + max(rem, 0))
            cursor = now + timedelta(minutes=max(rem, 0))
        elif j["status"] == "pending":
            item["est"] = est_minutes(n, history)
            if cursor is not None:
                cursor += timedelta(minutes=item["est"])
        timeline.append(item)

    return {
        "now": now.isoformat(timespec="seconds"),
        "current": current, "focus": focus, "focus_label": label_for(focus) if focus else None,
        "detail": detail, "epochs": epochs, "tail": log_tail(focus) if focus else [],
        "timeline": timeline,
        "queue_eta": cursor.isoformat(timespec="minutes") if cursor else None,
        "pending": sum(1 for t in timeline if t["status"] == "pending"),
        "runs": run_list,
        "dataset": dataset(),
        "gpu": gpu(),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def _json(self, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            try:
                return self._json(status())
            except Exception as e:  # never take the page down
                return self._json({"error": str(e)})
        if self.path.startswith("/api/jobs"):
            try:
                return self._json({"jobs": jobs(), "now": datetime.now().isoformat()})
            except Exception as e:
                return self._json({"error": str(e), "jobs": []})
        if self.path.startswith("/api/log/"):
            name = re.sub(r"[^\w.-]", "", self.path.rsplit("/", 1)[-1])
            return self._json({"name": name, "epochs": parse_epochs(name), "tail": log_tail(name, 40)})
        super().do_GET()

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Training monitor on http://127.0.0.1:{PORT}   (long jobs: http://127.0.0.1:{PORT}/jobs.html)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
