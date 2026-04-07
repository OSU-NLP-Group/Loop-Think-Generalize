import os
import argparse
import torch
import time
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import random, numpy as np
from transformers import GPT2Config
from gpt_utils_systematicity import CompositionDataset, CompositionTestDataset, load_vocab, custom_collate, \
    custom_collate_test, evaluate_model_test, RecurrentGPT2Block


def parse_args():
    parser = argparse.ArgumentParser(description="Train a RecurrentDepthTransformer model.")

    # Data paths
    parser.add_argument('--data_dir', type=str, default='data/composition.2000.200.7.2',
                        help='Path to dataset directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/systematicity/r4',
                        help='Path to save model checkpoints')
    parser.add_argument('--log_file', type=str, default='results/systematicity/r4.txt',
                        help='Path to save training logs')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=150001, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for optimizer')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Number of warmup steps for learning rate scheduler')

    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=768, help='Dimension of model embeddings')
    parser.add_argument('--num_recurrent_layers', type=int, default=4, help='Number of recurrent layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--recurrence', type=int, default=4, help='Number of recurrent iterations')

    parser.add_argument('--use_compile', action='store_true',
                        help='Enable torch.compile for the model')
    parser.add_argument('--compile_mode', type=str,
                        choices=['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
                        default='max-autotune',
                        help='torch.compile mode')

    # Precision settings
    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16'], default='bf16',
                        help='Enable mixed precision training (fp16 or bf16)')

    # Device configuration
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    return parser.parse_args()


def train_model(model, dataloader, valid_dataloader, test_dataloader, args):
    model.to(args.device)
    print(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_training_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    use_bf16 = args.precision == "bf16"
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model.train()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_time = time.time()
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", unit="batch")

        for input_ids, target_tokens, attention_mask, input_lengths in progress_bar:
            input_ids = input_ids.to(args.device)
            target_tokens = target_tokens.to(args.device)
            attention_mask = attention_mask.to(args.device)
            current_recurrence = args.recurrence
            model.num_iterations = current_recurrence
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=True):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                final_logits = logits[:, -1, :]

                loss = criterion(final_logits, target_tokens)

            if use_bf16:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss:.4f}")

        print("Running Evaluation...")

        model.num_iterations = args.recurrence
        test_acc_per_type = evaluate_model_test(model, test_dataloader, args.device)  # Get per-type accuracy
        test_acc_overall = sum(test_acc_per_type.values()) / len(test_acc_per_type) if test_acc_per_type else 0.0
        test_acc_str = ", ".join([f"{test_type}: {acc:.4f}" for test_type, acc in test_acc_per_type.items()])
        elapsed_seconds = time.time() - start_time
        with open(args.log_file, "a") as f:
            f.write(
                f"Epoch {epoch}: Test Acc: {test_acc_overall:.4f} Elapsed Time: {elapsed_seconds:.2f} ({test_acc_str})\n")

        if epoch % 10 == 0:
            if epoch > 100 or epoch % 1000 == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    vocab, vocab_size = load_vocab(os.path.join(args.data_dir, 'vocab.json'))
    train_dataset = CompositionDataset(os.path.join(args.data_dir, 'train.json'), vocab)
    valid_dataset = CompositionDataset(os.path.join(args.data_dir, 'valid.json'), vocab)
    test_dataset = CompositionTestDataset(os.path.join(args.data_dir, 'test.json'), vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=custom_collate_test)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=5,
        n_ctx=5,
        n_embd=args.d_model,
        n_layer=args.num_recurrent_layers,
        n_head=args.num_heads,
        _attn_implementation="eager",
    )

    # gpt2_block = GPT2Block(config)
    initial_recurrence = args.recurrence

    model = RecurrentGPT2Block(config, initial_recurrence)
    model = model.to(args.device)

    if args.use_compile:
        model = torch.compile(model, mode=args.compile_mode)

    print(f"Trainable parameters: {count_trainable_params(model)}")
    train_model(model, train_dataloader, valid_dataloader, test_dataloader, args)
