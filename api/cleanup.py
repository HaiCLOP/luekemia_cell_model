"""Delete opt-in uploads older than 24 hours from the private 'uploads' bucket. Run on a schedule
(e.g. hourly GitHub Action or cron) with SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the environment.

The bucket is listed page by page, so it keeps working past the 1,000-object page limit."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

PAGE = 1000


def expired(objects: list[dict], cutoff: datetime) -> list[str]:
    out = []
    for o in objects:
        ts = o.get("created_at")
        if ts and datetime.fromisoformat(ts.replace("Z", "+00:00")) < cutoff:
            out.append(o["name"])
    return out


def sweep(bucket, cutoff: datetime) -> tuple[int, int]:
    """Collect every expired name across all pages first, then delete, so offsets don't shift mid-scan."""
    scanned, old, offset = 0, [], 0
    while True:
        page = bucket.list("", {"limit": PAGE, "offset": offset, "sortBy": {"column": "name", "order": "asc"}})
        scanned += len(page)
        old += expired(page, cutoff)
        if len(page) < PAGE:
            break
        offset += PAGE
    for i in range(0, len(old), PAGE):
        bucket.remove(old[i:i + PAGE])
    return scanned, len(old)


def main() -> None:
    from supabase import create_client

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    scanned, removed = sweep(client.storage.from_("uploads"), cutoff)
    print(f"scanned {scanned} objects, removed {removed} older than 24 h")


if __name__ == "__main__":
    main()
