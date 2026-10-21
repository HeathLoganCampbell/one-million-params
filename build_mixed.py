"""Build one corpus from novels + research papers at a chosen share.

Mixing at raw sizes would bury the papers (0.57% of the prose), so the papers
are repeated to reach the target fraction. Repetition is the standard way to
up-weight a small domain; at 1M parameters the model cannot memorise the
repeated text, so it learns the style rather than the content.

    python build_mixed.py --share 0.10 --out data/mixed.txt
"""

import argparse
import math
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prose", default="data/gutenberg-full.txt")
    ap.add_argument("--papers", default="data/papers.txt")
    ap.add_argument("--share", type=float, default=0.10,
                    help="fraction of the final corpus that should be papers")
    ap.add_argument("--out", default="data/mixed.txt")
    args = ap.parse_args()

    with open(args.prose, "rb") as f:
        prose = f.read()
    with open(args.papers, "rb") as f:
        papers = f.read()

    if not 0 < args.share < 1:
        raise SystemExit("--share must be between 0 and 1")

    target_papers = args.share / (1 - args.share) * len(prose)
    repeat = max(1, math.ceil(target_papers / len(papers)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Interleave instead of appending: a trailing papers block would make the
    # validation split (the tail of the file) almost pure papers.
    with open(args.out, "wb") as out:
        span = len(prose) // repeat
        for i in range(repeat):
            out.write(prose[i * span : (i + 1) * span])
            out.write(b"\n\n")
            out.write(papers)
        out.write(prose[repeat * span :])

    total = len(prose) + repeat * (len(papers) + 2)
    print(f"prose      {len(prose) / 1e6:8.2f} MB")
    print(f"papers     {len(papers) / 1e6:8.2f} MB x {repeat} = {repeat * len(papers) / 1e6:.2f} MB")
    print(f"papers share {100 * repeat * len(papers) / total:7.2f}%")
    print(f"wrote {args.out}  ({total / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
