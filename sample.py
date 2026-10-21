"""Generate text from a trained checkpoint.

    python sample.py --ckpt checkpoints/model.pt --n 20
    python sample.py --ckpt checkpoints/model.pt --prompt "ROMEO:" --temperature 0.8
"""

import argparse

import torch

from model import GPT, GPTConfig


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/model.pt")
    p.add_argument("--n", type=int, default=10, help="number of samples")
    p.add_argument("--tokens", type=int, default=200, help="tokens per sample")
    p.add_argument("--prompt", default="", help="optional seed text")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"loaded {args.ckpt}  ({model.num_params():,} params, corpus={ckpt.get('corpus')})\n")

    seed = args.prompt.encode("utf-8") or b"\n"
    idx = torch.tensor(list(seed), dtype=torch.long, device=device)[None]

    for _ in range(args.n):
        out = model.generate(idx.clone(), args.tokens, args.temperature, args.top_k)
        # Drop the prompt itself; byte-level output can be invalid UTF-8, so replace.
        text = out[0].cpu().numpy().tobytes().decode("utf-8", errors="replace")
        print(text[len(args.prompt):].strip())
        print("-" * 60)


if __name__ == "__main__":
    main()
