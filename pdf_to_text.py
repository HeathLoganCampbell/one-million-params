"""Extract plain text from the PDFs in papers/ into a training corpus.

    python pdf_to_text.py                  # -> data/papers.txt
    python pdf_to_text.py --mix-with data/gutenberg-full.txt --out data/mixed.txt

PDF text extraction is lossy — expect running headers, page numbers, broken
hyphenation, and mangled equations.
"""

import argparse
import glob
import os
import re

from pypdf import PdfReader


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # re-join hyphenated words
    text = re.sub(r"\n{3,}", "\n\n", text)            # collapse blank runs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="papers")
    ap.add_argument("--out", default="data/papers.txt")
    ap.add_argument("--mix-with", default=None,
                    help="concatenate this extra corpus after the papers")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    if not files:
        raise SystemExit(f"no PDFs in {args.pdf_dir}/")

    chunks, total = [], 0
    print(f"{'file':58s} {'chars':>9s} {'pages':>6s}")
    for path in files:
        try:
            reader = PdfReader(path)
            raw = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            print(f"{os.path.basename(path):58s}  FAILED {e.__class__.__name__}")
            continue
        text = clean(raw)
        total += len(text)
        print(f"{os.path.basename(path):58s} {len(text):9,d} {len(reader.pages):6d}")
        chunks.append(f"\n\n=== {os.path.basename(path)} ===\n\n{text}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    body = "".join(chunks)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(body)
        if args.mix_with:
            with open(args.mix_with, "rb") as extra:
                f.write("\n\n")
                while chunk := extra.read(1 << 20):
                    f.write(chunk.decode("utf-8", errors="replace"))
            print(f"\nmixed in {args.mix_with}")

    size = os.path.getsize(args.out)
    print(f"\npapers text   {total:,} chars from {len(chunks)} files")
    print(f"wrote {args.out}  ({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
