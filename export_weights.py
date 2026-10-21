"""Dump checkpoint weights to the flat float32 layout the WASM kernel expects.

    python export_weights.py --ckpt checkpoints/gutenberg.pt --out web

Layout is documented at the top of web/assembly/index.ts.
"""

import argparse
import json
import os

import torch

from model import GPT, GPTConfig


def flatten(model: GPT) -> torch.Tensor:
    parts = [model.tok_emb.weight.detach()]
    for blk in model.blocks:
        parts += [
            blk.norm1.weight.detach(),
            blk.attn.qkv.weight.detach(),
            blk.attn.proj.weight.detach(),
            blk.norm2.weight.detach(),
            blk.mlp.w1.weight.detach(),
            blk.mlp.w2.weight.detach(),
            blk.mlp.w3.weight.detach(),
        ]
    parts.append(model.norm_f.weight.detach())
    return torch.cat([p.reshape(-1).float() for p in parts])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/gutenberg.pt")
    p.add_argument("--out", default="web")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    flat = flatten(model)
    expected = model.num_params()
    assert flat.numel() == expected, f"{flat.numel()} != {expected}"
    assert cfg.n_layer == 5 and cfg.n_embd == 128 and cfg.n_head == 4
    assert cfg.mlp_hidden == 336 and cfg.vocab_size == 256
    assert cfg.block_size <= 256, "kernel supports at most 256 positions"

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "model.bin")
    flat.numpy().astype("<f4").tofile(path)

    meta = {
        "params": int(flat.numel()),
        "dim": cfg.n_embd,
        "heads": cfg.n_head,
        "layers": cfg.n_layer,
        "mlp": cfg.mlp_hidden,
        "vocab": cfg.vocab_size,
        "block": cfg.block_size,
        "corpus": ckpt.get("corpus"),
    }
    with open(os.path.join(args.out, "model.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{args.ckpt} -> {path}  ({flat.numel():,} params, "
          f"{os.path.getsize(path) / 1e6:.2f} MB)")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
