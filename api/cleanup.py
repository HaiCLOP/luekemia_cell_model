"""Delete opt-in uploads older than 24 hours from the private 'uploads' bucket. Run on a schedule
(e.g. hourly GitHub Action or cron) with SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the environment."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from supabase import create_client


def main() -> None:
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    bucket = client.storage.from_("uploads")
    old = [o["name"] for o in bucket.list("", {"limit": 1000})
           if o.get("created_at") and datetime.fromisoformat(o["created_at"].replace("Z", "+00:00")) < cutoff]
    if old:
        bucket.remove(old)
    print(f"removed {len(old)} uploads older than 24 h")


if __name__ == "__main__":
    main()
