"""Build a large public-domain corpus from Project Gutenberg.

Reads Gutenberg's own catalogue CSV, keeps English works in the requested LoCC
classes (literature by default), downloads them into data/gutenberg/, strips the
licence boilerplate, and concatenates everything into one training file.

    python fetch_corpus.py --target-mb 200 --out data/gutenberg-full.txt
    python fetch_corpus.py --target-mb 1024 --locc P,Q,D,B --out data/gutenberg-1gb.txt

Cache is incremental: re-running only downloads what is missing.
"""

import argparse
import csv
import os
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import certifi

from data import _strip_gutenberg_boilerplate

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
BOOK_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
CTX = ssl.create_default_context(cafile=certifi.where())


def load_catalog(cache_path: str):
    """Download the catalogue once, then reuse it."""
    if not os.path.exists(cache_path + ".csv"):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        print(f"downloading catalogue -> {cache_path}.csv")
        with urllib.request.urlopen(CATALOG_URL, context=CTX, timeout=120) as r:
            with open(cache_path + ".csv", "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
    with open(cache_path + ".csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_books(rows, locc=("P",)):
    """English texts in the given LoCC classes, oldest Gutenberg IDs first."""
    out = []
    for r in rows:
        if r["Type"] != "Text" or r["Language"] != "en":
            continue
        if not any(r["LoCC"].startswith(p) for p in locc):  # P = language & literature
            continue
        try:
            bid = int(r["Text#"])
        except ValueError:
            continue
        out.append((bid, r["Title"][:70], r["Authors"][:40]))
    out.sort(key=lambda t: t[0])
    return out


def fetch(book_id: int, dest: str, tries: int = 4) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 5000:
        return "cached"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(BOOK_URL.format(id=book_id), context=CTX, timeout=45) as r:
                data = r.read()
            if len(data) < 5000:
                return "skip"
            with open(dest, "wb") as f:
                f.write(data)
            return "ok"
        except Exception as e:
            if attempt == tries - 1:
                return f"fail {e.__class__.__name__}"
            time.sleep(1.5 * (attempt + 1))
    return "fail"


def dir_size(paths):
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=float, default=200)
    ap.add_argument("--out", default="data/gutenberg-full.txt")
    ap.add_argument("--book-dir", default="data/gutenberg")
    ap.add_argument("--catalog", default="data/pg_catalog")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-books", type=int, default=4000)
    ap.add_argument(
        "--locc",
        default="P",
        help="comma-separated LoCC class prefixes to include (P=literature, Q=science, "
        "D=history, B=philosophy); default P",
    )
    args = ap.parse_args()

    locc = tuple(p.strip().upper() for p in args.locc.split(",") if p.strip())
    rows = load_catalog(args.catalog)
    books = pick_books(rows, locc)[: args.max_books]
    print(f"catalogue: {len(rows):,} books -> {len(books):,} english candidates in {locc}")

    os.makedirs(args.book_dir, exist_ok=True)
    target = args.target_mb * 1e6
    taken, stats = [], {}

    for start in range(0, len(books), 100):
        batch = books[start : start + 100]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fetch, bid, os.path.join(args.book_dir, f"{bid}.txt")): bid
                for bid, _, _ in batch
            }
            for fut in futures:
                result = fut.result()
                key = result.split()[0]
                stats[key] = stats.get(key, 0) + 1

        paths = [os.path.join(args.book_dir, f"{bid}.txt") for bid, _, _ in books]
        size = dir_size(paths)
        taken = [p for p in paths if os.path.exists(p) and os.path.getsize(p) > 5000]
        print(f"  {len(taken):5d} books  {size / 1e6:7.1f} MB  {stats}")
        if size >= target:
            break

    print(f"\nconcatenating {len(taken)} books")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for p in taken:
            with open(p, "rb") as f:
                text = f.read().decode("utf-8", errors="replace")
            text = _strip_gutenberg_boilerplate(text)
            out.write(text)
            out.write("\n\n")
            total += len(text)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB, {total:,} chars)")


if __name__ == "__main__":
    main()
