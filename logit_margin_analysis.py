import os
import re
import json
import math
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from transformers import GPT2Config
from tqdm import tqdm

from gpt_utils_extrapolation import (
    load_vocab,
    CompositionTestDataset,
    custom_collate_test,
    RecurrentGPT2Block,
)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_tag(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip().lower())


def make_attn_4d(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    attn = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=dtype)
    return (1.0 - attn) * -10000.0


@torch.no_grad()
def embed_like_model(model: RecurrentGPT2Block, input_ids: torch.Tensor) -> torch.Tensor:
    bsz, seq_len = input_ids.size()
    device = input_ids.device
    tok = model.token_embedding(input_ids)

    if model.positional_embedding_type == "learned":
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        h = tok + model.position_embedding(pos_ids)
    elif model.positional_embedding_type == "sinusoidal":
        h = tok + model.pe[:seq_len, :].unsqueeze(0)
    else:
        h = tok
    return model.dropout(h)


def clamp_first_pad_pos_from_input_lengths(input_lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.clamp(input_lengths.long(), 0, T - 1)


def split_to_display_label(split_name: str) -> str:
    m = re.match(r"(\d+)hop_test", split_name)
    if m:
        return f"{m.group(1)}-hop"
    return split_name

@torch.no_grad()
def eval_split_margin(
    model: RecurrentGPT2Block,
    dataloader,
    iters: int,
    device: torch.device,
    split_name: str
) -> Dict[str, Any]:
    model.eval()

    sum_margin = np.zeros((iters,), dtype=np.float64)
    n_total = 0
    first_hit_iters_all = []

    for batch in tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False):
        input_ids, target_tokens, attention_mask, input_lengths = batch[:4]
        input_ids = input_ids.to(device)
        target_tokens = target_tokens.to(device)
        attention_mask = attention_mask.to(device)
        input_lengths = torch.as_tensor(input_lengths, device=device, dtype=torch.long)

        B, T_seq = input_ids.shape
        n_total += B

        monitor_pos = clamp_first_pad_pos_from_input_lengths(input_lengths, T_seq)
        ar = torch.arange(B, device=device)

        h = embed_like_model(model, input_ids)
        init_emb = h.clone()
        attn_4d = make_attn_4d(attention_mask, dtype=h.dtype)

        t_first = torch.full((B,), -1, device=device, dtype=torch.long)

        for t in range(iters):
            if getattr(model, "input_injection", False) and t > 0:
                h = h + init_emb

            for blk in model.blocks:
                h = blk(h, attention_mask=attn_4d)[0]

            v_all = model.ln_f(h)
            logits = model.lm_head(v_all)[ar, monitor_pos, :].float()

            target_logit = logits[ar, target_tokens]
            masked_logits = logits.clone()
            masked_logits[ar, target_tokens] = -float("inf")
            competitor_logit = torch.max(masked_logits, dim=1)[0]

            sum_margin[t] += float((target_logit - competitor_logit).sum().item())

            top1 = torch.argmax(logits, dim=1)
            hit_now = (top1 == target_tokens) & (t_first == -1)
            if hit_now.any():
                t_first = torch.where(hit_now, torch.full_like(t_first, t), t_first)

        hit_mask = (t_first != -1)
        if hit_mask.any():
            first_hit_iters_all.append((t_first[hit_mask] + 1).detach().cpu().numpy())

    denom = max(n_total, 1)
    avg_t_first = float(np.concatenate(first_hit_iters_all).mean()) if first_hit_iters_all else float("nan")

    return {
        "count": int(n_total),
        "avg_t_first": avg_t_first,
        "margin": (sum_margin / denom).astype(np.float32),
    }


def load_single_model(
    checkpoint_path: str,
    config: GPT2Config,
    iters: int,
    device: torch.device,
    pos_type: str,
    trans_type: str,
    inj: bool,
) -> RecurrentGPT2Block:
    model = RecurrentGPT2Block(
        config,
        num_iterations=iters,
        positional_embedding_type=pos_type,
        transformer_type=trans_type,
        input_injection=inj,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = {
        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
        for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def save_results_json(path: str, results: Dict[str, Any]):
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(results), f, indent=2)


def load_results_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def compute_global_ylim(results: Dict[str, Any], model_order: List[str]) -> Tuple[float, float]:
    vals = []
    for model_key in model_order:
        if model_key not in results["models"]:
            continue
        for split_name, split_res in results["models"][model_key]["splits"].items():
            vals.extend(split_res["margin"])

    vals = np.array(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return -1.0, 1.0

    y_min = float(vals.min())
    y_max = float(vals.max())

    pad = 0.08 * max(1e-6, (y_max - y_min))
    return y_min - pad, y_max + pad


def plot_margin_grid(
    results: Dict[str, Any],
    out_png: str,
    out_pdf: str,
    title: str = "",
):
    model_order = results["meta"]["model_order"]
    split_order = results["meta"]["split_order"]
    num_iters = results["meta"]["iters"]
    x_iter = np.arange(1, num_iters + 1)

    cmap = plt.get_cmap("tab10")
    color_map = {split: cmap(i % 10) for i, split in enumerate(split_order)}

    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.2, 6.2),
        dpi=220,
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    y_min, y_max = compute_global_ylim(results, model_order)

    for ax, model_key in zip(axes, model_order):
        model_res = results["models"][model_key]
        panel_title = model_res["label"]

        for split_name in split_order:
            if split_name not in model_res["splits"]:
                continue

            split_res = model_res["splits"][split_name]
            y = np.array(split_res["margin"], dtype=np.float32)
            avg_t_first = float(split_res["avg_t_first"])
            color = color_map[split_name]

            ax.plot(
                x_iter, y,
                linewidth=1.8,
                color=color,
                alpha=0.95
            )

            if math.isfinite(avg_t_first):
                ax.axvline(
                    x=avg_t_first,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                    color=color
                )

        ax.axhline(
            y=0.0,
            color="black",
            linestyle="-",
            linewidth=1.0,
            alpha=0.75
        )

        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.set_xlim(1, num_iters)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis="both", labelsize=8)

    fig.supxlabel("Recurrent Iteration (t)", fontsize=10, y=0.08)
    fig.supylabel("Average Logit Margin (Target - Top Competitor)", fontsize=10, x=0.03)

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    split_handles = [
        Line2D([0], [0], color=color_map[s], lw=2.2, label=split_to_display_label(s))
        for s in split_order
    ]
    extra_handles = [
        Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="Avg. first-hit iteration $t^*$"),
        Line2D([0], [0], color="black", lw=1.0, linestyle="-", label="Decision boundary (margin = 0)"),
    ]
    handles = split_handles + extra_handles

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
        fontsize=8
    )

    plt.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.90,
        bottom=0.22,
        wspace=0.18,
        hspace=0.25
    )

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser("4-model margin grid analysis")

    p.add_argument("--checkpoint_root", type=str,
                   default="checkpoints/multi_hop")
    p.add_argument("--data_dir", type=str,
                   default="data/multi_hop")
    p.add_argument("--test_file", type=str, default="test.json")
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--num_recurrent_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--positional_embedding_type", type=str, default="none")
    p.add_argument("--transformer_type", type=str, default="regular")
    p.add_argument("--input_injection", action="store_true")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--sample_size", type=int, default=0, help="0 = full split")

    p.add_argument("--out_dir", type=str, default="outputs/paper_margin")
    p.add_argument("--plot_dir", type=str, default="plots/paper_margin")
    p.add_argument("--results_json", type=str, default="")
    p.add_argument("--figure_tag", type=str, default="margin_grid_4models")
    p.add_argument("--title", type=str,
                   default="Margin vs Recurrent Iteration Across Fixed and Dynamic Recurrence Models")
    p.add_argument("--replot_only", action="store_true")

    args = p.parse_args()

    set_seed(42)
    ensure_dir(args.out_dir)
    ensure_dir(args.plot_dir)

    if args.results_json.strip():
        results_json_path = args.results_json
    else:
        results_json_path = os.path.join(args.out_dir, "margin_grid_results.json")

    out_png = os.path.join(args.plot_dir, f"{safe_tag(args.figure_tag)}.png")
    out_pdf = os.path.join(args.plot_dir, f"{safe_tag(args.figure_tag)}.pdf")

    if args.replot_only:
        print(f"Loading cached results from: {results_json_path}")
        results = load_results_json(results_json_path)
        plot_margin_grid(results, out_png, out_pdf, title=args.title)
        print(f"Saved figure to:\n  {out_png}\n  {out_pdf}")
        return

    device = torch.device(args.device)

    vocab, vocab_size = load_vocab(os.path.join(args.data_dir, "vocab.json"))
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.max_len,
        n_ctx=args.max_len,
        n_embd=args.d_model,
        n_layer=args.num_recurrent_layers,
        n_head=args.num_heads,
        _attn_implementation="eager",
    )

    # Requested models / checkpoints
    model_specs = [
        {
            "key": "r6",
            "label": "R=6",
            "dir_name": "r6",
            "checkpoint_epoch": 1015,
            "splits": ["2hop_test", "5hop_test", "10hop_test", "13hop_test", "16hop_test"],
        },
        {
            "key": "r7",
            "label": "R=7",
            "dir_name": "r7",
            "checkpoint_epoch": 1050,
            "splits": ["2hop_test", "5hop_test", "10hop_test", "13hop_test", "16hop_test"],
        },
        {
            "key": "r8",
            "label": "R=8",
            "dir_name": "r8",
            "checkpoint_epoch": 1259,
            "splits": ["2hop_test", "5hop_test", "10hop_test", "13hop_test", "16hop_test"],
        },
        {
            "key": "rdyn",
            "label": "R=dynamic",
            "dir_name": "r_dyn",
            "checkpoint_epoch": 5388,
            "splits": ["2hop_test", "5hop_test", "10hop_test", "13hop_test", "16hop_test", "30hop_test"],
        },
    ]

    split_order = ["2hop_test", "5hop_test", "10hop_test", "13hop_test", "16hop_test", "30hop_test"]

    results = {
        "meta": {
            "checkpoint_root": args.checkpoint_root,
            "data_dir": args.data_dir,
            "test_file": args.test_file,
            "iters": args.iters,
            "batch_size": args.batch_size,
            "sample_size": args.sample_size,
            "max_len": args.max_len,
            "positional_embedding_type": args.positional_embedding_type,
            "transformer_type": args.transformer_type,
            "input_injection": bool(args.input_injection),
            "model_order": [m["key"] for m in model_specs],
            "split_order": split_order,
        },
        "models": {}
    }

    for spec in model_specs:
        checkpoint_path = os.path.join(
            args.checkpoint_root,
            spec["dir_name"],
            f"checkpoint_epoch_{spec['checkpoint_epoch']}.pt"
        )

        print(f"\nLoading model {spec['label']} from:\n  {checkpoint_path}")
        model = load_single_model(
            checkpoint_path=checkpoint_path,
            config=config,
            iters=args.iters,
            device=device,
            pos_type=args.positional_embedding_type,
            trans_type=args.transformer_type,
            inj=args.input_injection,
        )

        model_res = {
            "label": spec["label"],
            "checkpoint_path": checkpoint_path,
            "checkpoint_epoch": spec["checkpoint_epoch"],
            "splits": {}
        }

        for split in spec["splits"]:
            full_ds = CompositionTestDataset(args.data_dir, args.test_file, vocab, [split])

            if len(full_ds) == 0:
                print(f"  [Warning] Split '{split}' is empty. Skipping.")
                continue

            if args.sample_size > 0 and len(full_ds) > args.sample_size:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(full_ds), size=args.sample_size, replace=False).tolist()
                ds = Subset(full_ds, indices)
            else:
                ds = full_ds

            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda b: custom_collate_test(b, args.max_len),
            )

            res = eval_split_margin(model, dl, args.iters, device, split)
            model_res["splits"][split] = {
                "count": int(res["count"]),
                "avg_t_first": float(res["avg_t_first"]),
                "margin": res["margin"].tolist(),
            }

            print(
                f"  {split:12s} | n={res['count']:4d} | "
                f"avg_t_first={res['avg_t_first']:.2f}"
            )

        results["models"][spec["key"]] = model_res

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_results_json(results_json_path, results)
    print(f"\nSaved raw results to:\n  {results_json_path}")

    plot_margin_grid(results, out_png, out_pdf, title=args.title)
    print(f"Saved figure to:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()