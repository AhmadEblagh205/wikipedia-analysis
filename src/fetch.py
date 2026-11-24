"""Fetch Wikipedia summary extracts through Wikimedia's REST endpoint.

The repository analyses short summary extracts rather than full article bodies.
Fetched content is intentionally kept out of git; a local manifest records the
source URL, page id/revision timestamp when supplied by Wikimedia, retrieval
time and the exact API URL used.

Usage:
    python src/fetch.py "Battery (electricity)" Catalysis Probability
    python src/fetch.py "Methanol" --delay 1.0
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
DEFAULT_USER_AGENT = "wikipedia-analysis/0.3 (https://www.ahmadnabileblagh.com/)"


def slug(title: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    if not value:
        raise ValueError(f"Title does not produce a safe filename: {title!r}")
    return value


def api_url(title: str) -> str:
    normalised = title.strip().replace(" ", "_")
    if not normalised:
        raise ValueError("Article title must not be empty")
    return API.format(title=quote(normalised, safe=""))


def fetch_summary(title: str, user_agent: str = DEFAULT_USER_AGENT) -> dict:
    """Fetch one summary and return text plus provenance metadata."""
    if not user_agent.strip():
        raise ValueError("A descriptive User-Agent is required")

    url = api_url(title)
    response = requests.get(
        url,
        headers={"User-Agent": user_agent, "Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    extract = payload.get("extract")
    if not isinstance(extract, str) or not extract.strip():
        raise ValueError(f"Wikipedia returned no usable summary extract for {title!r}")

    content_urls = payload.get("content_urls") or {}
    desktop = content_urls.get("desktop") or {}
    return {
        "requested_title": title,
        "resolved_title": payload.get("title", title),
        "extract": extract.strip(),
        "pageid": payload.get("pageid"),
        "revision_timestamp": payload.get("timestamp"),
        "source_url": desktop.get("page"),
        "api_url": url,
    }


def _validate_delay(value: str) -> float:
    delay = float(value)
    if delay < 0:
        raise argparse.ArgumentTypeError("delay must be non-negative")
    return delay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("titles", nargs="+", help="English Wikipedia article titles")
    parser.add_argument(
        "--delay",
        type=_validate_delay,
        default=1.0,
        help="seconds between serial requests (default: 1.0)",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="descriptive User-Agent including operator contact information",
    )
    parser.add_argument("--out-dir", default=RAW)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slugs = [slug(title) for title in args.titles]
    if len(slugs) != len(set(slugs)):
        raise SystemExit("Requested titles collide after filename slugging; rename the inputs")

    manifest = []
    for index, (title, file_slug) in enumerate(zip(args.titles, slugs)):
        record = fetch_summary(title, user_agent=args.user_agent)
        destination = output_dir / f"{file_slug}.txt"
        destination.write_text(record.pop("extract"), encoding="utf-8")
        record.update(
            {
                "file": destination.name,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        manifest.append(record)
        print(f"{title}: -> {destination}")
        if index < len(args.titles) - 1 and args.delay:
            time.sleep(args.delay)

    (output_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest -> {output_dir / '_manifest.json'}")


if __name__ == "__main__":
    main()
