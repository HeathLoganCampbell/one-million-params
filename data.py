"""Corpus loading and batching. Byte-level, so no tokenizer is needed."""

import os
import re
import shutil
import ssl
import urllib.request

import numpy as np
import torch

try:  # macOS python.org builds ship without a usable CA bundle
    import certifi

    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    _SSL_CONTEXT = None

CORPORA = {
    # ~228 KB of first names, one per line
    "names": (
        "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
        "names.txt",
    ),
    # ~1.1 MB of Shakespeare
    "shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "shakespeare.txt",
    ),
}

# Public-domain novels from Project Gutenberg (~12 MB total).
GUTENBERG_BOOKS = [
    (84, "Frankenstein"),
    (2701, "Moby Dick"),
    (345, "Dracula"),
    (1661, "The Adventures of Sherlock Holmes"),
    (1342, "Pride and Prejudice"),
    (11, "Alice's Adventures in Wonderland"),
    (16, "Peter Pan"),
    (98, "A Tale of Two Cities"),
    (120, "Treasure Island"),
    (113, "The Secret Garden"),
    (1400, "Great Expectations"),
    (768, "Wuthering Heights"),
    (1260, "Jane Eyre"),
    (36, "The War of the Worlds"),
    (35, "The Time Machine"),
    (174, "The Picture of Dorian Gray"),
    (164, "Twenty Thousand Leagues under the Sea"),
    (18857, "A Journey to the Centre of the Earth"),
    (62, "A Princess of Mars"),
    (78, "Tarzan of the Apes"),
    (64, "The Gods of Mars"),
    (219, "Heart of Darkness"),
    (5200, "The Metamorphosis"),
    (1952, "The Yellow Wallpaper"),
    (64317, "The Great Gatsby"),
]


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Drop the license header/footer so the model only sees the story."""
    m = re.search(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*", text, re.I
    )
    if m:
        text = text[m.end() :]
    m = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text, re.I)
    if m:
        text = text[: m.start()]
    return text.strip()


def load_gutenberg(data_dir: str = "data", ids=None) -> str:
    """Download and concatenate a shelf of public-domain novels."""
    root = os.path.join(data_dir, "gutenberg")
    os.makedirs(root, exist_ok=True)
    books = [(i, t) for i, t in GUTENBERG_BOOKS if ids is None or i in ids]
    parts = []
    for book_id, title in books:
        dest = os.path.join(root, f"{book_id}.txt")
        if not os.path.exists(dest):
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            print(f"downloading {title} ({book_id})")
            with urllib.request.urlopen(url, context=_SSL_CONTEXT) as resp:
                with open(dest, "wb") as f:
                    shutil.copyfileobj(resp, f)
        with open(dest, "rb") as f:
            raw = f.read().decode("utf-8", errors="replace")
        parts.append(_strip_gutenberg_boilerplate(raw))
    return "\n\n".join(parts)


def load_corpus(name: str = "names", data_dir: str = "data", path: str | None = None) -> str:
    """Return the corpus as a string, downloading it if necessary."""
    if path is not None:
        with open(path, "rb") as f:
            return f.read().decode("utf-8")

    if name == "gutenberg":
        return load_gutenberg(data_dir)

    if name == "gutenberg-full":
        path = os.path.join(data_dir, "gutenberg-full.txt")
        if not os.path.exists(path):
            raise SystemExit(
                "data/gutenberg-full.txt not found - run: python fetch_corpus.py --target-mb 200"
            )
        with open(path, "rb") as f:
            return f.read().decode("utf-8")

    if name == "papers":
        path = os.path.join(data_dir, "papers.txt")
        if not os.path.exists(path):
            raise SystemExit(
                "data/papers.txt not found - run: python pdf_to_text.py"
            )
        with open(path, "rb") as f:
            return f.read().decode("utf-8")

    if name == "mixed":
        path = os.path.join(data_dir, "mixed.txt")
        if not os.path.exists(path):
            raise SystemExit(
                "data/mixed.txt not found - run: python build_mixed.py"
            )
        with open(path, "rb") as f:
            return f.read().decode("utf-8")

    if name not in CORPORA:
        raise SystemExit(f"unknown corpus {name!r}; choose from {list(CORPORA)} or pass --data PATH")

    url, filename = CORPORA[name]
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, filename)
    if not os.path.exists(dest):
        print(f"downloading {name} -> {dest}")
        with urllib.request.urlopen(url, context=_SSL_CONTEXT) as resp:
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
    with open(dest, "rb") as f:
        return f.read().decode("utf-8")


def encode(text: str) -> np.ndarray:
    """UTF-8 bytes as a uint8 array. One token per byte."""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def train_val_split(data: np.ndarray, val_frac: float = 0.1):
    n = int(len(data) * (1.0 - val_frac))
    return data[:n], data[n:]


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    """Random contiguous windows: x is the input, y is x shifted by one byte."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
