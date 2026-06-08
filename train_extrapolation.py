import os
import argparse
import torch
from tqdm import tqdm
import time
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import random, numpy as np
from functools import partial
from transformers import GPT2Config
from gpt_utils_extrapolation import CompositionDataset, CompositionTestDataset, load_vocab, custom_collate, \
    custom_collate_test, evaluate_model_test, evaluate_model_test_adaptive, RecurrentGPT2Block
import re
import json

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Recurrent-Depth Transformer model for multi-hop composition.")

    parser.add_argument('--data_dir', type=str, default='data/multi_hop/',
                        help='Path to dataset directory')
    parser.add_argument('--test_file', type=str, default='test.json',
                        help='Path to dataset directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/multi_hop/r_dyn/',
                        help='Path to save model checkpoints')
    parser.add_argument('--log_file', type=str,
                        default='results/multi_hop/r_dyn.txt',
                        help='Path to save training logs')

    parser.add_argument('--num_epochs', type=int, default=100001, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum number of tokens')
    parser.add_argument('--max_hop', type=int, default=40, help='Maximum number of hops')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--acc_threshold_curriculum', type=float, default=0.95,
                        help='Threshold beyond which we move to next stage in curriculum training.')
    parser.add_argument('--pred_pos', type=str, choices=['inp_len', 'last_token'], default="last_token",
                        help='Position for output prediction.')

    parser.add_argument('--resume_training', action='store_true',
                        help='Resume curriculum from a checkpoint and skip completed hop stages.')

    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Checkpoint filename inside --checkpoint_dir OR an absolute path.')

    parser.add_argument('--hops_generalized', type=int, default=9,
                        help='Last hop level already completed (e.g., 10). Training resumes at hops_generalized+1.')

    parser.add_argument('--use_lr_decay', action='store_true',
                        help='lr decay after each stage.')
    parser.add_argument('--stage_lr_base', type=float, default=1e-4,
                        help='LR for the first curriculum stage (2-hop).')
    parser.add_argument('--stage_lr_gamma', type=float, default=0.9,
                        help='Multiply LR by this factor each new stage.')
    parser.add_argument('--stage_lr_min', type=float, default=2e-5,
                        help='Min LR for any curriculum stage.')
    parser.add_argument('--max_grad_norm', type=float, default=0.0,
                        help='Maximum gradient norm for gradient clipping. Set to 0.0 to disable.')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--c_scale', type=float, default=0.0)
    parser.add_argument('--input_injection', action='store_true',
                        help='Enable input injection (adding input embeddings at the start of each recurrence).')
    parser.add_argument('--force_grok', action='store_true',
                        help='Force 1000 epochs on 2-hop curriculum stage.')

    parser.add_argument('--train_mode', type=str, choices=['max-autotune', 'regular'], default='max-autotune',
                        help='Whether training run was optimized.')

    parser.add_argument('--d_model', type=int, default=768, help='Dimension of model embeddings')
    parser.add_argument('--num_recurrent_layers', type=int, default=4, help='Number of recurrent layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--recurrence', type=int, default=1, help='Number of recurrent iterations')

    parser.add_argument('--recurrence_type', type=str, choices=['fixed', 'dynamic'], default='dynamic',
                        help='Type of recurrence during training. "fixed" uses --recurrence. "dynamic" uses Poisson distribution.')

    parser.add_argument('--dynamic_min', type=int, default=2, help='Min iterations when R=dynamic')
    parser.add_argument('--dynamic_max', type=int, default=8, help='Max iterations when R=dynamic')
    parser.add_argument('--dynamic_mean', type=int, default=4, help='Mean iterations when R=dynamic')

    parser.add_argument('--positional_embedding_type', type=str, choices=['learned', 'sinusoidal', 'none'],
                        default='none',
                        help='Type of positional embedding to use: learned, sinusoidal, or none (NoPE).')

    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16'], default='bf16',
                        help='Enable mixed precision training (fp16 or bf16)')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    # Wandb arguments
    parser.add_argument('--use_wandb', default=False, action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='loopreason',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity/team name')

    return parser.parse_args()


def count_hops(input_text):
    """Count number of relations (hops) in input_text."""
    token_pattern = re.compile(r'<[^>]+>')
    tokens = token_pattern.findall(input_text)
    # First token is entity, rest are relations
    return len(tokens) - 1


def train_model(model, dataloader, test_dataloader, args, test_type, start_epoch=0, start_time=time.time(),
                hop_count=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_training_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

    use_bf16 = args.precision == "bf16"
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    model.train()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    current_epoch = start_epoch
    while True:
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch + 1}", unit="batch")

        for input_ids, target_tokens, attention_mask, input_lengths in progress_bar:

            if args.recurrence_type == 'dynamic':
                dynamic_rec = np.random.poisson(args.dynamic_mean)
                dynamic_rec = max(args.dynamic_min, min(args.dynamic_max, dynamic_rec))
                model.num_iterations = dynamic_rec
            else:
                model.num_iterations = args.recurrence

            input_ids = input_ids.to(args.device)
            target_tokens = target_tokens.to(args.device)
            attention_mask = attention_mask.to(args.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=autocast_dtype, enabled=True):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if args.pred_pos == "last_token":
                    logits = outputs.logits[:, -1, :]
                else:
                    idx = (input_lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.logits.size(-1)).to(args.device)
                    logits = outputs.logits.gather(1, idx).squeeze(1)
                loss = criterion(logits, target_tokens)

            if use_bf16:
                loss.backward()
                if args.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

            # Log batch-level metrics to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': current_epoch + 1,
                    'test_type': test_type,
                    'hop_count': hop_count if hop_count is not None else 0
                })

        avg_loss = total_loss / len(dataloader)

        if args.recurrence_type == 'dynamic':
            model.num_iterations = args.dynamic_max
            test_acc_per_type = evaluate_model_test(model, test_dataloader, args.device, pred_pos=args.pred_pos)
            test_acc_overall = sum(test_acc_per_type.values()) / len(test_acc_per_type) if test_acc_per_type else 0.0
        else:
            model.num_iterations = args.recurrence
            test_acc_per_type = evaluate_model_test(model, test_dataloader, args.device, pred_pos=args.pred_pos)
            test_acc_overall = sum(test_acc_per_type.values()) / len(test_acc_per_type) if test_acc_per_type else 0.0
        elapsed_seconds = time.time() - start_time

        test_acc_str = ", ".join([f"{test_type}: {acc:.4f}" for test_type, acc in test_acc_per_type.items()])

        with open(args.log_file, "a") as f:
            f.write(
                f"Epoch {current_epoch + 1}: Elapsed Time: {elapsed_seconds:.2f} Test Acc: {test_acc_overall:.4f} ({test_acc_str})\n")

        # Log epoch-level metrics to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'train/epoch_loss': avg_loss,
                'test/overall_accuracy': test_acc_overall,
                'test/current_test_type_accuracy': test_acc_per_type[test_type],
                'train/epoch': current_epoch + 1,
                'train/elapsed_time': elapsed_seconds,
                'test/current_test_type': test_type,
                'curriculum/hop_count': hop_count if hop_count is not None else 0
            }
            # Log per-type test accuracies
            for acc_type, acc_value in test_acc_per_type.items():
                log_dict[f'test/accuracy_{acc_type}'] = acc_value
            wandb.log(log_dict)

        split_acc = test_acc_per_type[test_type]

        # Break if max epochs reached
        if current_epoch >= args.num_epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{current_epoch + 1}.pt")
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

            # Log checkpoint to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'checkpoint/saved': True,
                    'checkpoint/epoch': current_epoch + 1,
                    'checkpoint/test_accuracy': split_acc,
                    'checkpoint/path': checkpoint_path,
                    'checkpoint/reason': 'max_epochs_reached'
                })
            break

        to_break = False
        if split_acc > args.acc_threshold_curriculum:
            if not args.force_grok:
                to_break = True
            else:
                if current_epoch >= 1000:
                    to_break = True
        if to_break:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{current_epoch + 1}.pt")
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

            # Log checkpoint to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'checkpoint/saved': True,
                    'checkpoint/epoch': current_epoch + 1,
                    'checkpoint/test_accuracy': split_acc,
                    'checkpoint/path': checkpoint_path
                })

            break
        current_epoch += 1
    print('Done generalizing on ' + test_type + ' after ' + str(current_epoch) + ' epochs')
    return current_epoch + 1


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    vocab, vocab_size = load_vocab(os.path.join(args.data_dir, 'vocab.json'))

    # Initialize wandb if enabled
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: --use_wandb specified but wandb is not installed. Install with: pip install wandb")
            args.use_wandb = False
        else:
            # Create wandb config from args
            wandb_config = {
                'data_dir': args.data_dir,
                'test_file': args.test_file,
                'seed': args.seed,
                'batch_size': args.batch_size,
                'max_len': args.max_len,
                'max_hop': args.max_hop,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'warmup_steps': args.warmup_steps,
                'acc_threshold_curriculum': args.acc_threshold_curriculum,
                'd_model': args.d_model,
                'num_recurrent_layers': args.num_recurrent_layers,
                'num_heads': args.num_heads,
                'recurrence': args.recurrence,
                'recurrence_type': args.recurrence_type,
                'dynamic_min': args.dynamic_min,
                'dynamic_max': args.dynamic_max,
                'dynamic_mean': args.dynamic_mean,
                'positional_embedding_type': args.positional_embedding_type,
                'precision': args.precision,
                'device': args.device,
            }

            wandb.init(
                project=args.wandb_project,
                name=args.checkpoint_dir,
                entity=args.wandb_entity,
                config=wandb_config,
                reinit=True
            )
            print(f"Wandb initialized: project={args.wandb_project}, run={wandb.run.name}")

    # Load train.json once from 50hop_nonoverlap directory
    train_json_path = os.path.join(args.data_dir, 'train.json')
    print(f"Loading training data from {train_json_path}...")
    with open(train_json_path, 'r') as f:
        all_train_data = json.load(f)
    print(f"Loaded {len(all_train_data)} training instances")

    # Log dataset info to wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({'dataset/total_training_instances': len(all_train_data)})

    # Generate test types for each hop count
    test_types = []
    for i in range(2, args.max_hop + 1):
        test_types.append(str(i) + 'hop_test')

    config = GPT2Config(vocab_size=vocab_size, n_positions=args.max_len, n_ctx=args.max_len, n_embd=args.d_model,
                        n_layer=args.num_recurrent_layers, n_head=args.num_heads, embd_pdrop=args.dropout,
                        _attn_implementation="eager")

    model = RecurrentGPT2Block(config, args.recurrence, positional_embedding_type=args.positional_embedding_type,
                               input_injection=args.input_injection, c_scale=args.c_scale)
    model.to(args.device)

    if args.resume_training:
        if args.resume_checkpoint is None:
            raise ValueError("--resume_training requires --resume_checkpoint")
        if args.hops_generalized < 2:
            raise ValueError("--hops_generalized must be >= 2 when resuming (e.g., 10).")

        ckpt_path = args.resume_checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(args.checkpoint_dir, ckpt_path)

        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device)

        if args.train_mode == 'regular':
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
        else:
            sd = ckpt["model_state_dict"]

            if any(k.startswith("_orig_mod.") for k in sd.keys()):
                sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
            model.load_state_dict(sd, strict=True)

        resumed_epoch = int(ckpt.get("epoch", 0))
        current_epoch = resumed_epoch + 1
    else:
        current_epoch = 0

    if args.train_mode != 'regular':
        model = torch.compile(model, mode=args.train_mode)

    # current_epoch = 0
    all_valid_test = []
    start_time = time.time()
    if args.resume_training:
        # pre-fill splits already completed: ['2hop_test', ..., f'{hops_generalized}hop_test']
        all_valid_test = [f"{i}hop_test" for i in range(2, args.hops_generalized + 1)]
        start_k = args.hops_generalized + 1
    else:
        start_k = 2

    if start_k > args.max_hop:
        print(f"Nothing to do: start_k={start_k} > max_hop={args.max_hop}")

    else:
        for k in range(start_k, args.max_hop + 1):
            test_type = f"{k}hop_test"

            # Filter data to include all instances with hop count <= k
            filtered_data = [item for item in all_train_data if count_hops(item['input_text']) <= k]
            print(f"Hop count {k}: Using {len(filtered_data)} training instances (all hops <= {k})")

            # Log curriculum stage info to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'curriculum/stage': k,
                    'curriculum/training_instances': len(filtered_data),
                    'curriculum/test_type': test_type
                })

            train_dataset = CompositionDataset(vocab=vocab, data=filtered_data)
            all_valid_test.append(test_type)
            test_dataset = CompositionTestDataset(args.data_dir, args.test_file, vocab, all_splits=all_valid_test)
            custom_collate_test_fn = partial(custom_collate_test, max_len=args.max_len)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                         collate_fn=custom_collate_test_fn)

            custom_collate_fn = partial(custom_collate, max_len=args.max_len)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=custom_collate_fn)

            if args.use_lr_decay:
                args.lr = max(args.stage_lr_base * (args.stage_lr_gamma ** (k - 2)), args.stage_lr_min)
                print(f"[Curriculum] Stage {k}: lr={args.lr:.2e}")
            current_epoch = train_model(model, train_dataloader, test_dataloader, args, test_type,
                                        start_epoch=current_epoch, start_time=start_time, hop_count=k)

        # Save final checkpoint at the end of training
        final_checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
        print(f"Saving final checkpoint to {final_checkpoint_path}...")
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'loss': None,  # Final loss not available at this point
        }, final_checkpoint_path)
        print(f"Final checkpoint saved at epoch {current_epoch}")

        # Log final checkpoint to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'checkpoint/final_saved': True,
                'checkpoint/final_epoch': current_epoch,
                'checkpoint/final_path': final_checkpoint_path
            })

        # Finish wandb run
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
