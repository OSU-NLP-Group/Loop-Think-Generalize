import os
import argparse
import torch
import json
from torch.utils.data import DataLoader
import random, numpy as np
from functools import partial
from transformers import GPT2Config
from gpt_utils_extrapolation import CompositionTestDataset, load_vocab, custom_collate_test, evaluate_model_test, \
    RecurrentGPT2Block


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a RecurrentDepthTransformer model across a range of recurrence depths.")

    parser.add_argument('--data_dir', type=str, default='data/multi_hop')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/multi_hop/r_dyn/')
    parser.add_argument('--model_name', type=str, default='checkpoint_epoch_5388.pt')
    parser.add_argument('--output_file', type=str,
                        default='outputs/multi_hop/r_dyn.json',
                        help='Path to save the output JSON file with results.')
    parser.add_argument('--input_injection', action='store_true',
                        help='Enable input injection (adding input embeddings at the start of each recurrence).')
    parser.add_argument('--pred_pos', type=str, choices=['inp_len', 'last_token'], default="last_token",
                        help='Whether training run was optimized.')

    parser.add_argument(
        "--recurrence_range",
        type=int,
        nargs="+",
        # default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        help="List of recurrence depths"
    )

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--max_hop', type=int, default=40)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_recurrent_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--positional_embedding_type', type=str, default='none')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    vocab, vocab_size = load_vocab(os.path.join(args.data_dir, 'vocab.json'))
    all_splits = [f"{i}hop_test" for i in range(2, args.max_hop + 1)]
    test_dataset = CompositionTestDataset(args.data_dir, args.test_file, vocab, all_splits=all_splits)
    custom_collate_test_fn = partial(custom_collate_test, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=custom_collate_test_fn)

    config = GPT2Config(vocab_size=vocab_size, n_positions=args.max_len, n_ctx=args.max_len, n_embd=args.d_model,
                        n_layer=args.num_recurrent_layers, n_head=args.num_heads, _attn_implementation="eager")

    model = RecurrentGPT2Block(config, 1, positional_embedding_type=args.positional_embedding_type,
                               input_injection=args.input_injection)

    device = torch.device(args.device)
    checkpoint_path = os.path.join(args.checkpoint_dir, args.model_name)
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["model_state_dict"]

    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}

    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    all_results = {}

    recurrence_range = args.recurrence_range
    for recurrence_depth in recurrence_range:
        print(f"\n--- Evaluating with Recurrence = {recurrence_depth} ---")
        model.num_iterations = recurrence_depth
        test_acc_per_type = evaluate_model_test(model, test_dataloader, device, pred_pos=args.pred_pos)
        all_results[recurrence_depth] = test_acc_per_type
        print(f"Results for {recurrence_depth} iterations: {test_acc_per_type}")

    print(f"\nSweep complete. Saving all results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print("Done.")
