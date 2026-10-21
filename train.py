"""Train the ~1M parameter byte-level transformer.

    python train.py --corpus names --steps 3000
    python train.py --corpus shakespeare --steps 5000
    python train.py --data /path/to/your.txt --steps 10000
"""

import argparse
import math
import os
import time

import torch

from data import encode, get_batch, load_corpus, train_val_split
from model import GPT, GPTConfig


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_optimizer(model, lr, weight_decay):
    # No weight decay on norms and any 1-D parameters.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def lr_at(step, args):
    """Linear warmup then cosine decay to min_lr."""
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    if step >= args.steps:
        return args.min_lr
    ratio = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss(model, splits, batch_size, block_size, device, iters=50):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="names",
                   choices=["names", "shakespeare", "gutenberg", "gutenberg-full", "papers", "mixed"])
    p.add_argument("--data", default=None, help="path to your own text file (overrides --corpus)")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=5)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from --out if it exists")
    p.add_argument("--init-from", default=None, metavar="CKPT",
                   help="load weights from CKPT but start a fresh schedule at step 0")
    p.add_argument("--export-web", default=None, metavar="DIR",
                   help="also write DIR/model.bin at every checkpoint (live site updates)")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()

    text = load_corpus(args.corpus, path=args.data)
    data = encode(text)
    train_data, val_data = train_val_split(data)
    splits = {"train": train_data, "val": val_data}

    cfg = GPTConfig(
        vocab_size=256,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)

    print(f"device       {device}")
    print(f"corpus       {args.data or args.corpus}  ({len(data):,} bytes)")
    print(f"config       {cfg.n_layer}L {cfg.n_head}H {cfg.n_embd}d  "
          f"mlp={cfg.mlp_hidden}  block={cfg.block_size}")
    print(f"parameters   {model.num_params():,}  (non-embedding {model.num_params(True):,})")
    print(f"tokens/step  {args.batch_size * args.block_size:,}")
    print()

    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    start_step = 0
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"initialised weights from {args.init_from} (step counter reset)")
    elif args.resume and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"resumed {args.out} at step {start_step}")

    def save_checkpoint():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": vars(cfg),
                "corpus": args.data or args.corpus,
            },
            tmp,
        )
        os.replace(tmp, args.out)  # atomic, safe to read while training continues

        if args.export_web:
            from export_weights import flatten

            os.makedirs(args.export_web, exist_ok=True)
            dest = os.path.join(args.export_web, "model.bin")
            flat = flatten(model).detach().cpu().numpy().astype("<f4")
            flat.tofile(dest + ".tmp")
            os.replace(dest + ".tmp", dest)

    t0 = time.time()
    tokens_seen = 0
    step = start_step
    for step in range(start_step, args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)

        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        tokens_seen += x.numel()

        if step % 50 == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            tps = tokens_seen / max(elapsed, 1e-9)
            print(f"step {step:5d}  loss {loss.item():.4f}  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                  f"{tps:,.0f} tok/s  {elapsed:6.1f}s")

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            losses = estimate_loss(model, splits, args.batch_size, args.block_size,
                                   device, args.eval_iters)
            print(f"          eval: train {losses['train']:.4f}  val {losses['val']:.4f}")

        if args.sample_every and step > 0 and step % args.sample_every == 0:
            # newline, not NUL: token 0 is out of distribution and produces noise
            start = torch.tensor([[10]], dtype=torch.long, device=device)
            out = model.generate(start, 200, temperature=0.9, top_k=40)
            print("          ---")
            print("          " + out[0].cpu().numpy().tobytes()
                  .decode("utf-8", errors="replace").replace("\n", "\n          "))
            print("          ---")

        if args.save_every and step > 0 and step % args.save_every == 0:
            save_checkpoint()

    losses = estimate_loss(model, splits, args.batch_size, args.block_size,
                           device, args.eval_iters)
    print(f"\nfinal: train {losses['train']:.4f}  val {losses['val']:.4f}  "
          f"in {time.time() - t0:.1f}s")

    save_checkpoint()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
