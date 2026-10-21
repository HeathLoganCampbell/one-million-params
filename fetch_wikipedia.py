"""Download the most-viewed Wikipedia articles as a plain-text corpus.

Pulls the daily top-viewed list from the Wikimedia pageviews API, drops
non-article namespaces (Main Page, Special:, Portal:, ...), then fetches the
plain-text extract of each article through the MediaWiki API.

    python fetch_wikipedia.py                 # latest day with data, top 20
    python fetch_wikipedia.py --top 500 --date 2026/10/20
    python fetch_wikipedia.py --out data/wiki-top20.txt

Writes a manifest (rank, title, views, chars) next to the corpus.
"""

import argparse
import datetime as dt
import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request

import certifi

CTX = ssl.create_default_context(cafile=certifi.where())
UA = "tiny-gpt-corpus/1.0 (local research; text corpus build)"
TOP_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{y}/{m}/{d}"
EXTRACT_URL = "https://en.wikipedia.org/w/api.php"

# Namespaces and utility pages that dominate the top-viewed list but are not prose.
SKIP_PREFIXES = (
    "Main_Page", "Special:", "Wikipedia:", "Portal:", "File:", "Help:",
    "Talk:", "Template:", "Category:", "Draft:", "Module:", "MediaWiki:",
    "Book:", "TimedText:", "wikipedia:", "WIKIPEDIA:",
)
SKIP_TITLES = {"Main Page", "-", "Main_Page", "Wikipedia"}


def get_json(url: str, params: dict | None = None, tries: int = 4):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, context=CTX, timeout=45) as r:
                return json.load(r)
        except Exception as e:
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def top_articles(date: dt.date, limit: int) -> list[dict]:
    """Top-viewed article titles for one day, filtered to real articles."""
    url = TOP_URL.format(y=date.year, m=f"{date.month:02d}", d=f"{date.day:02d}")
    data = get_json(url)
    out = []
    for rank, item in enumerate(data["items"][0]["articles"], start=1):
        title = urllib.parse.unquote(item["article"]).replace("_", " ")
        if title in SKIP_TITLES or title.startswith(SKIP_PREFIXES):
            continue
        if ":" in title.split(" ")[0]:  # any other namespace prefix
            continue
        out.append({"rank": rank, "title": title, "views": item["views"]})
        if len(out) >= limit:
            break
    return out


def extract(title: str) -> str:
    data = get_json(
        EXTRACT_URL,
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
        },
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        return ""
    return pages[0].get("extract", "") or ""


def latest_day_with_data(start: dt.date, back: int = 14) -> dt.date:
    for i in range(back):
        day = start - dt.timedelta(days=i)
        url = TOP_URL.format(y=day.year, m=f"{day.month:02d}", d=f"{day.day:02d}")
        try:
            get_json(url, tries=1)
            return day
        except Exception:
            continue
    raise SystemExit(f"no pageview data in the {back} days before {start}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20, help="how many articles")
    ap.add_argument("--date", default=None, help="YYYY/MM/DD (default: latest available)")
    ap.add_argument("--out", default="data/wikipedia-top20.txt")
    ap.add_argument("--manifest", default=None, help="default: <out>.json")
    args = ap.parse_args()

    if args.date:
        day = dt.datetime.strptime(args.date, "%Y/%m/%d").date()
    else:
        day = latest_day_with_data(dt.date.today())

    articles = top_articles(day, args.top)
    print(f"{day}: fetching {len(articles)} articles")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    manifest = []
    total = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for a in articles:
            text = re.sub(r"\n{3,}", "\n\n", extract(a["title"])).strip()
            if not text:
                print(f"  {a['rank']:3d}  {a['title']!r}: no extract, skipped")
                continue
            out.write(f"\n\n=== {a['title']} ===\n\n")
            out.write(text)
            total += len(text)
            manifest.append({**a, "chars": len(text)})
            print(f"  {a['rank']:3d}  {a['views']:>9,}  {len(text):>7,} chars  {a['title']}")

    manifest_path = args.manifest or args.out + ".json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"date": str(day), "articles": manifest}, f, indent=2)

    print(f"\nwrote {args.out}  ({total:,} chars) and {manifest_path}")


if __name__ == "__main__":
    main()
