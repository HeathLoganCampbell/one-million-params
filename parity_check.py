"""Dump reference logits from PyTorch for the WASM parity test."""

import argparse
import json

import torch

from model import GPT, GPTConfig

PROMPT = "It was the best of times, it was the"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/model.pt")
    p.add_argument("--out", default="web/expected.json")
    p.add_argument("--prompt", default=PROMPT)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = torch.tensor(list(args.prompt.encode("utf-8")), dtype=torch.long)[None]
    assert ids.shape[1] <= cfg.block_size, f"prompt too long for block {cfg.block_size}"

    with torch.no_grad():
        logits, _ = model(ids)
    last = logits[0, -1].tolist()

    with open(args.out, "w") as f:
        json.dump({"prompt": args.prompt, "tokens": ids[0].tolist(), "logits": last}, f)
    print(f"wrote {args.out}: {len(last)} logits for {ids.shape[1]} tokens")


if __name__ == "__main__":
    main()
