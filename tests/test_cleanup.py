"""Retention sweep: every page is scanned, only expired objects are removed."""
from datetime import datetime, timedelta, timezone

from api import cleanup


class FakeBucket:
    def __init__(self, objects):
        self.objects = sorted(objects, key=lambda o: o["name"])
        self.removed = []

    def list(self, _path, opts):
        return self.objects[opts["offset"]: opts["offset"] + opts["limit"]]

    def remove(self, names):
        assert len(names) <= cleanup.PAGE
        self.removed += names


def test_sweep_covers_every_page(monkeypatch):
    monkeypatch.setattr(cleanup, "PAGE", 10)
    now = datetime.now(timezone.utc)
    old = (now - timedelta(hours=30)).isoformat().replace("+00:00", "Z")
    new = (now - timedelta(hours=1)).isoformat()
    objs = [{"name": f"{i:03d}.png", "created_at": old if i % 2 else new} for i in range(35)]
    b = FakeBucket(objs)
    scanned, removed = cleanup.sweep(b, now - timedelta(hours=24))
    assert scanned == 35 and removed == 17
    assert sorted(b.removed) == [o["name"] for o in objs if o["created_at"] == old]


def test_objects_without_timestamp_are_kept():
    now = datetime.now(timezone.utc)
    assert cleanup.expired([{"name": "folder"}], now) == []
