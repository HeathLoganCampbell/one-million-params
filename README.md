# tiny-gpt — a 1M parameter transformer that runs in WebAssembly

A byte-level decoder-only transformer with **1,006,976 parameters** — small
enough to train in minutes on a laptop, and small enough to run entirely in your
browser from a **13 KB WebAssembly module** and a **4 MB weight file**.

There is no server and no API call. The forward pass — embeddings, 5 layers of
RMSNorm → QKV → RoPE → causal attention → SwiGLU, tied output head — is
compiled from AssemblyScript to wasm and executed on the client.

> **It is a text-completion model, not a chatbot.** It continues what you type;
> it does not answer questions. Nothing in this pipeline includes instruction
> tuning.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# train a small model (~1 min)
.venv/bin/python train.py --corpus names --steps 3000 --block-size 128

# run the site
.venv/bin/python export_weights.py --ckpt checkpoints/model.pt --out web
cd web && npm install && npm run build && npm run serve
# -> http://localhost:8080
```

## Layout

| Path | |
|---|---|
| `model.py` | the transformer |
| `data.py` | corpora, byte encoding, batching |
| `fetch_corpus.py` | builds a Gutenberg corpus from the official catalogue |
| `pdf_to_text.py` | turns `papers/*.pdf` into a training corpus |
| `build_mixed.py` | mixes corpora at a chosen weight |
| `train.py` | training loop, cosine LR, eval, checkpointing |
| `export_weights.py` | flat float32 weights for the wasm kernel |
| `parity_check.py` | PyTorch reference logits for kernel verification |
| `web/assembly/index.ts` | the inference kernel |
| `web/` | static site |

## Verified

The wasm kernel is checked against PyTorch on identical input:

```
max |diff|  2.646e-5
argmax      py=121 wasm=121  MATCH
```

## Training

```bash
# first names, ~1 min
.venv/bin/python train.py --corpus names --steps 3000 --block-size 128

# Shakespeare, ~10 min
.venv/bin/python train.py --corpus shakespeare --steps 5000

# 25 public-domain novels (~12.8 MB) from Project Gutenberg
.venv/bin/python train.py --corpus gutenberg --steps 4000 --dropout 0.1

# a large shelf: 500+ novels (~237 MB) built by fetch_corpus.py
.venv/bin/python fetch_corpus.py --target-mb 200
.venv/bin/python train.py --corpus gutenberg-full --steps 8000 --dropout 0.1

# 17 papers from papers/*.pdf (~1.3 MB) - a domain specialist
.venv/bin/python pdf_to_text.py
.venv/bin/python train.py --corpus papers --steps 2500 --dropout 0.1

# your own text file
.venv/bin/python train.py --data /path/to/corpus.txt --steps 10000
```

## Sample

```bash
.venv/bin/python sample.py --ckpt checkpoints/model.pt --n 10
.venv/bin/python sample.py --ckpt checkpoints/model.pt --prompt "ROMEO:" --temperature 0.8
```

## Run it in the browser (WebAssembly)

The inference kernel in `web/assembly/index.ts` is AssemblyScript, compiled to a
13 KB wasm module that holds the whole forward pass. Weights are a flat 4 MB
float32 blob.

```bash
cd web
npm install

# build the wasm kernel
npm run build

# export weights from a checkpoint
cd .. && .venv/bin/python export_weights.py --ckpt checkpoints/gutenberg.pt --out web && cd web

# serve
npm run serve        # http://localhost:8080
```

Verify the kernel against PyTorch at any time:

```bash
cd .. && .venv/bin/python parity_check.py --ckpt checkpoints/gutenberg.pt && cd web
npm test             # compares wasm logits to PyTorch's
```

The site is a text-completion interface, not a chatbot — the model has no
instruction tuning, so it continues your prompt rather than answering it.

## The shape of the model

| | |
|---|---|
| Layers | 5 |
| Heads | 4 |
| Width | 128 |
| MLP hidden | 336 (SwiGLU) |
| Context | 256 bytes |
| Vocab | 256 (UTF-8 bytes) |
| Parameters | **1,006,976** |

Parameter budget, per layer:

```
attention   4·d²      = 65,536
SwiGLU      3·d·h     = 129,024
                      ---------
per layer              194,560
× 5 layers             972,800
embeddings (tied) 256·128  32,768
norms                          1,408
                      ---------
total                1,006,976
```

## Changing the size

```bash
# half a million -- 3 layers
python train.py --n-layer 3

# ~2.5M -- 10 layers
python train.py --n-layer 10
```

Because the vocabulary is bytes, parameter count depends only on
`n_layer`, `n_embd`, `n_head`, and `mlp_hidden` — never on the corpus.
Note the `12·L·d²` rule: halving width quarters the parameter count, so
width is a much sharper knob than depth.

## Files

```
model.py           architecture (RMSNorm, RoPE, SwiGLU, tied embeddings)
data.py            corpus download, byte encoding, batching
fetch_corpus.py    builds a large corpus from Gutenberg's catalogue
pdf_to_text.py     extracts papers/*.pdf into a training corpus
train.py           training loop, cosine LR, eval, sampling
sample.py          load a checkpoint and generate
export_weights.py  dump flat float32 weights for the WASM kernel
parity_check.py    reference logits from PyTorch, for testing the kernel
web/               AssemblyScript kernel + static site
```

## Domain specialists

A 1M-parameter model learns one distribution well. Mixing corpora of very
different sizes does not give you a model that does both — it gives you the
larger domain with noise. The papers corpus is 0.57% of the novels corpus, so
the way to use it is a second model:

```bash
python pdf_to_text.py                              # 17 PDFs -> data/papers.txt
python train.py --corpus papers --steps 2500 --dropout 0.1 \
    --export-web web/models/papers
```

The site loads `web/models/<name>/model.bin` and exposes a switcher in the
header, so both specialists share one WASM kernel.

## Notes

- `--block-size 128` is enough for names; use 256+ for prose.
- Watch the gap between train and val loss. If val starts rising while train
  falls, you're memorizing — cut steps or add `--dropout 0.1`.
- At this size, data quality matters more than every hyperparameter combined.
