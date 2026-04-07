import os
import re
import json
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


class CompositionDataset(Dataset):
    def __init__(self, vocab, data):
        self.data = data
        self.vocab = vocab
        self.token_pattern = re.compile(r'<[^>]+>')

    def tokenize(self, text):
        return self.token_pattern.findall(text)

    def tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        input_tokens = self.tokenize(sample['input_text'])
        target_tokens = self.tokenize(sample['target_text'])

        # input_tokens.append('<sep>')

        if target_tokens and target_tokens[-1] == "</a>":
            target_tokens = target_tokens[:-1]

        input_ids = torch.tensor(self.tokens_to_ids(input_tokens), dtype=torch.long)
        target_ids = torch.tensor(self.tokens_to_ids(target_tokens), dtype=torch.long)

        return input_ids, target_ids


class CompositionTestDataset(Dataset):
    def __init__(self, data_dir, file, vocab, all_splits=None):
        self.data = []
        with open(os.path.join(data_dir, file), 'r') as f:
            data = json.load(f)
        if all_splits:
            for sample in data:
                if sample['type'] in all_splits:
                    self.data.append(sample)
        else:
            self.data = data
        self.vocab = vocab
        self.token_pattern = re.compile(r'<[^>]+>')

    def tokenize(self, text):
        return self.token_pattern.findall(text)

    def tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        input_tokens = self.tokenize(sample['input_text'])
        target_tokens = self.tokenize(sample['target_text'])
        test_type = sample['type']

        # input_tokens.append('<sep>')

        if target_tokens and target_tokens[-1] == "</a>":
            target_tokens = target_tokens[:-1]

        input_ids = torch.tensor(self.tokens_to_ids(input_tokens), dtype=torch.long)
        target_ids = torch.tensor(self.tokens_to_ids(target_tokens), dtype=torch.long)

        return input_ids, target_ids, test_type


def custom_collate(batch, max_len=25):
    input_ids_list, target_ids_list = zip(*batch)
    pad_idx = 0

    padded_input_ids = []
    input_lengths = []
    for ids in input_ids_list:
        length = len(ids)
        input_lengths.append(min(length, max_len))
        if length < max_len:
            padded = torch.cat([ids, torch.full((max_len - length,), pad_idx, dtype=torch.long)])
        else:
            padded = ids[:max_len]
        padded_input_ids.append(padded)

    padded_input_ids = torch.stack(padded_input_ids)

    # padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_idx)
    target_tokens = torch.tensor([x[-1].item() for x in target_ids_list], dtype=torch.long)
    attention_mask = (padded_input_ids != pad_idx).long()
    input_lengths = torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.long)

    return padded_input_ids, target_tokens, attention_mask, input_lengths


def custom_collate_test(batch, max_len=25):
    input_ids_list, target_ids_list, test_type = zip(*batch)
    pad_idx = 0

    padded_input_ids = []
    input_lengths = []
    for ids in input_ids_list:
        length = len(ids)
        input_lengths.append(min(length, max_len))
        if length < max_len:
            padded = torch.cat([ids, torch.full((max_len - length,), pad_idx, dtype=torch.long)])
        else:
            padded = ids[:max_len]
        padded_input_ids.append(padded)

    padded_input_ids = torch.stack(padded_input_ids)

    # padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_idx)
    target_tokens = torch.tensor([x[-1].item() for x in target_ids_list], dtype=torch.long)
    attention_mask = (padded_input_ids != pad_idx).long()
    input_lengths = torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.long)

    return padded_input_ids, target_tokens, attention_mask, input_lengths, test_type


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab_list = json.load(f)
    # If the pad token is not present, add it at the beginning.
    if "<pad>" not in vocab_list:
        vocab_list = ["<pad>"] + vocab_list
    vocab = {token: idx for idx, token in enumerate(vocab_list)}
    return vocab, len(vocab)


def evaluate_model_test_adaptive(model, dataloader, device, pred_pos="last_token", max_recurrence=16, pad_idx=0):
    model.eval()

    correct_per_type = defaultdict(int)
    total_per_type = defaultdict(int)

    old_num_iterations = getattr(model, "num_iterations", None)
    model.num_iterations = max_recurrence

    with torch.no_grad():
        for batch in dataloader:
            input_ids, target_tokens, attention_mask, input_lengths, test_types = batch

            input_ids = input_ids.to(device)
            target_tokens = target_tokens.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model.forward_adapative_ent(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_lengths=input_lengths,
                pred_pos=pred_pos
            )
            output_logits = outputs["final_logits"]

            if pred_pos == "last_token":
                logits = output_logits[:, -1, :]
            else:
                idx = (input_lengths - 1).view(-1, 1, 1).expand(-1, 1, output_logits.size(-1)).to(device)
                logits = output_logits.gather(1, idx).squeeze(1)

            predicted_tokens = torch.argmax(logits, dim=-1)

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                test_type = test_types[i]
                if target_tokens[i] != pad_idx:
                    is_correct = (predicted_tokens[i] == target_tokens[i])
                    correct_per_type[test_type] += int(is_correct)
                    total_per_type[test_type] += 1

    if old_num_iterations is not None:
        model.num_iterations = old_num_iterations

    accuracy_per_type = {
        tt: correct_per_type[tt] / total_per_type[tt]
        for tt in total_per_type if total_per_type[tt] > 0
    }
    return accuracy_per_type


def evaluate_model_test(model, dataloader, device, pred_pos="last_token", pad_idx=0):
    model.eval()

    correct_per_type = defaultdict(int)
    total_per_type = defaultdict(int)

    with torch.no_grad():
        for batch in dataloader:
            input_ids, target_tokens, attention_mask, input_lengths, test_types = batch

            input_ids = input_ids.to(device)
            target_tokens = target_tokens.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if pred_pos == "last_token":
                logits = outputs.logits[:, -1, :]
            else:
                idx = (input_lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.logits.size(-1)).to(device)
                logits = outputs.logits.gather(1, idx).squeeze(1)

            predicted_tokens = torch.argmax(logits, dim=-1)

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                test_type = test_types[i]
                if target_tokens[i] != pad_idx:
                    is_correct = (predicted_tokens[i] == target_tokens[i])
                    correct_per_type[test_type] += int(is_correct)
                    total_per_type[test_type] += 1

    accuracy_per_type = {tt: correct_per_type[tt] / total_per_type[tt]
                         for tt in total_per_type if total_per_type[tt] > 0}
    return accuracy_per_type


def evaluate_model_test_adaptive_metrics(
        model,
        dataloader,
        device,
        max_recurrence,
        pred_pos="last_token",
        eps_kl=0.001,
        entropy_thresh=3.00,
        min_iters=1,
):
    model.eval()
    old_num_iterations = getattr(model, "num_iterations", None)
    model.num_iterations = max_recurrence

    correct_per_type = defaultdict(int)
    total_per_type = defaultdict(int)
    iterations_per_type = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating with Adaptive Recurrence (KL + Entropy)"):
            if len(batch) == 6:
                input_ids, target_tokens, attention_mask, input_lengths, test_types, _ = batch
            else:
                input_ids, target_tokens, attention_mask, input_lengths, test_types = batch

            input_ids = input_ids.to(device)
            target_tokens = target_tokens.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model.forward_adapative(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_lengths=input_lengths,
                pred_pos=pred_pos,
                eps_kl=eps_kl,
                entropy_thresh=entropy_thresh,
                min_iters=min_iters,
            )

            final_logits = outputs["final_logits"]
            iterations_used = outputs["iterations_used"]

            if pred_pos == "inp_len":
                input_lengths_dev = torch.as_tensor(input_lengths, device=device, dtype=torch.long)
                current_predict_pos = (input_lengths_dev - 1).clamp(min=0, max=final_logits.size(1) - 1)
                pred_ids = final_logits[
                           torch.arange(final_logits.size(0), device=device), current_predict_pos, :
                           ].argmax(dim=-1)
            else:
                pred_ids = final_logits[:, -1, :].argmax(dim=-1)

            for i in range(input_ids.size(0)):
                tt = test_types[i]
                tgt_id = target_tokens[i].item()

                iterations_per_type[tt].append(int(iterations_used[i].item()))
                correct_per_type[tt] += int(pred_ids[i].item() == tgt_id)
                total_per_type[tt] += 1

    if old_num_iterations is not None:
        model.num_iterations = old_num_iterations

    accuracy_per_type = {
        tt: (correct_per_type[tt] / total_per_type[tt]) * 100.0
        for tt in total_per_type if total_per_type[tt] > 0
    }

    avg_iterations_per_type = {
        tt: float(np.mean(v)) for tt, v in iterations_per_type.items()
    }

    iterations_stats_per_type = {}
    for tt, lst in iterations_per_type.items():
        arr = np.array(lst, dtype=np.float32)
        iterations_stats_per_type[tt] = {
            "count": int(arr.size),
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "median": float(np.median(arr)) if arr.size else float("nan"),
            "std": float(arr.std(ddof=0)) if arr.size else float("nan"),
            "min": float(arr.min()) if arr.size else float("nan"),
            "max": float(arr.max()) if arr.size else float("nan"),
            "values": lst,
        }

    return accuracy_per_type, avg_iterations_per_type, iterations_stats_per_type


class RecurrentGPT2Block(nn.Module):
    def __init__(self, config, num_iterations, positional_embedding_type='learned', input_injection=False,
                 c_scale=0.003):
        super().__init__()
        self.config = config
        self.num_iterations = num_iterations
        self.input_injection = input_injection
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_embedding_type = positional_embedding_type

        if self.positional_embedding_type == 'learned':
            self.position_embedding = nn.Embedding(config.n_positions, config.n_embd)
        elif self.positional_embedding_type == 'sinusoidal':
            self.position_embedding = None
            pe = torch.zeros(config.n_positions, config.n_embd)
            position = torch.arange(0, config.n_positions, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10000.0) / config.n_embd))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        elif self.positional_embedding_type == 'none':
            self.position_embedding = None
        else:
            raise ValueError(f"Unsupported positional embedding type: {self.positional_embedding_type}")

        embd_pdrop = getattr(config, "embd_pdrop", 0.1)
        self.dropout = nn.Dropout(embd_pdrop)

        layer_norm_epsilon = getattr(config, "layer_norm_epsilon", 1e-5)
        self.ln_f = nn.LayerNorm(config.n_embd, eps=layer_norm_epsilon)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        scale = c_scale
        for block in self.blocks:
            block.attn.c_proj.weight.data.mul_(scale)
            block.mlp.c_proj.weight.data.mul_(scale)
            if block.attn.c_proj.bias is not None:
                block.attn.c_proj.bias.data.zero_()
            if block.mlp.c_proj.bias is not None:
                block.mlp.c_proj.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        token_embeds = self.token_embedding(input_ids)

        if self.positional_embedding_type == 'learned':
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
            pos_embeds = self.position_embedding(position_ids)
            hidden_states = token_embeds + pos_embeds
        elif self.positional_embedding_type == 'sinusoidal':
            pos_embeds = self.pe[:seq_len, :].unsqueeze(0)
            hidden_states = token_embeds + pos_embeds
        else:
            hidden_states = token_embeds

        hidden_states = self.dropout(hidden_states)
        initial_embeddings = hidden_states

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        for i in range(self.num_iterations):
            if self.input_injection and i > 0:
                hidden_states = hidden_states + initial_embeddings
            for block in self.blocks:
                hidden_states = block(hidden_states, attention_mask=attention_mask)[0]

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return type("Output", (object,), {"logits": logits})

    def forward_adapative_ent(self, input_ids, attention_mask, input_lengths, pred_pos="last_token", H0=3.0, H1=12.0,
                              eps_kl=0.5, min_iters=1):

        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        token_embeds = self.token_embedding(input_ids)
        if self.positional_embedding_type == 'learned':
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
            pos_embeds = self.position_embedding(position_ids)
            hidden_states = token_embeds + pos_embeds
        elif self.positional_embedding_type == 'sinusoidal':
            pos_embeds = self.pe[:seq_len, :].unsqueeze(0)
            hidden_states = token_embeds + pos_embeds
        else:
            hidden_states = token_embeds
        hidden_states = self.dropout(hidden_states)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if pred_pos == "inp_len":
            monitor_idx = (input_lengths - 1).view(-1, 1, 1).expand(-1, 1, self.config.vocab_size).to(device)
        else:
            monitor_idx = torch.full((batch_size, 1, 1), seq_len - 1, dtype=torch.long, device=device).expand(-1, 1,
                                                                                                              self.config.vocab_size)

        max_iters = self.num_iterations

        stable = torch.zeros(batch_size, dtype=torch.bool, device=device)
        iteration_counts = torch.full((batch_size,), max_iters - 1, dtype=torch.long, device=device)

        final_logits = None
        chosen_logits = None

        H_thresh = (H0 + (H1 / float(seq_len))) * torch.ones(batch_size, device=device)
        logp_prev = None
        first_kl_iter = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        first_entropy_iter = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        for t in range(max_iters):
            for block in self.blocks:
                hidden_states = block(hidden_states, attention_mask=extended_attention_mask)[0]

            all_logits_t = self.lm_head(self.ln_f(hidden_states))
            final_logits = all_logits_t
            logits_at_pos = all_logits_t.gather(1, monitor_idx).squeeze(1)
            logp_t = F.log_softmax(logits_at_pos, dim=-1)
            p_t = logp_t.exp()

            if t > 0:
                kl = torch.sum(p_t * (logp_t - logp_prev), dim=-1)
                entropy = -torch.sum(p_t * logp_t, dim=-1)

                new_kl = (kl < eps_kl) & (first_kl_iter == -1)
                first_kl_iter[new_kl] = t
                new_entropy = (entropy < H_thresh) & (first_entropy_iter == -1)
                first_entropy_iter[new_entropy] = t

                ok_min = torch.tensor(t + 1 >= min_iters, device=device)
                newly_stable = (kl < eps_kl) & (entropy < H_thresh) & ~stable & ok_min

                if newly_stable.any():
                    if chosen_logits is None:
                        chosen_logits = torch.empty_like(all_logits_t)
                    chosen_logits[newly_stable] = all_logits_t[newly_stable]
                    iteration_counts[newly_stable] = t
                    stable |= newly_stable

            logp_prev = logp_t
            if stable.all():
                break

        if chosen_logits is None:
            chosen_logits = final_logits
        else:
            not_stable = ~stable
            if not_stable.any():
                chosen_logits[not_stable] = final_logits[not_stable]

        iterations_used = iteration_counts + 1
        halt_driver = torch.full((batch_size,), 2, dtype=torch.long, device=device)

        if max_iters > 1:
            big = max_iters + 10
            t_kl_eff = torch.where(first_kl_iter >= 0, first_kl_iter, torch.full_like(first_kl_iter, big))
            t_ent_eff = torch.where(first_entropy_iter >= 0, first_entropy_iter,
                                    torch.full_like(first_entropy_iter, big))
            stacked = torch.stack([t_kl_eff, t_ent_eff], dim=1)
            _, argmax_idx = stacked.max(dim=1)
            halted_early = stable
            halt_driver[halted_early] = argmax_idx[halted_early]

        return {
            "final_logits": chosen_logits,
            "iteration_counts": iteration_counts,
            "iterations_used": iterations_used,
            "halt_driver": halt_driver.detach().cpu()
        }

    def forward_adapative(
            self,
            input_ids,
            attention_mask,
            input_lengths,
            pred_pos="last_token",
            eps_kl=0.01,
            entropy_thresh=3.00,
            min_iters=1,
    ):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        token_embeds = self.token_embedding(input_ids)

        if self.positional_embedding_type == 'learned':
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
            pos_embeds = self.position_embedding(position_ids)
            hidden_states = token_embeds + pos_embeds
        elif self.positional_embedding_type == 'sinusoidal':
            pos_embeds = self.pe[:seq_len, :].unsqueeze(0)
            hidden_states = token_embeds + pos_embeds
        else:
            hidden_states = token_embeds

        hidden_states = self.dropout(hidden_states)
        initial_embeddings = hidden_states

        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(hidden_states.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        input_lengths = torch.as_tensor(input_lengths, device=device, dtype=torch.long)

        if pred_pos == "inp_len":
            monitor_pos = (input_lengths - 1).clamp(min=0, max=seq_len - 1)
        else:
            monitor_pos = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=device)

        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        iteration_counts = torch.full((batch_size,), self.num_iterations - 1, dtype=torch.long, device=device)

        chosen_logits = None
        final_logits = None
        logp_prev = None

        for t in range(self.num_iterations):
            if self.input_injection and t > 0:
                hidden_states = hidden_states + initial_embeddings

            for block in self.blocks:
                hidden_states = block(hidden_states, attention_mask=extended_attention_mask)[0]

            all_logits_t = self.lm_head(self.ln_f(hidden_states))
            final_logits = all_logits_t

            logits_at_pos = all_logits_t[
                            torch.arange(batch_size, device=device), monitor_pos, :
                            ]
            logp_t = F.log_softmax(logits_at_pos, dim=-1)

            if t > 0:
                p_t = logp_t.exp()
                kl = torch.sum(p_t * (logp_t - logp_prev), dim=-1)
                entropy = -torch.sum(p_t * logp_t, dim=-1)

                new_halts = (
                        (~halted)
                        & ((t + 1) >= min_iters)
                        & (kl < eps_kl)
                        & (entropy < entropy_thresh)
                )

                if new_halts.any():
                    if chosen_logits is None:
                        chosen_logits = torch.empty_like(all_logits_t)
                    chosen_logits[new_halts] = all_logits_t[new_halts]
                    iteration_counts[new_halts] = t
                    halted |= new_halts

                if halted.all():
                    break

            logp_prev = logp_t

        if chosen_logits is None:
            chosen_logits = final_logits
        else:
            not_halted = ~halted
            if not_halted.any():
                chosen_logits[not_halted] = final_logits[not_halted]

        return {
            "final_logits": chosen_logits,
            "iteration_counts": iteration_counts,
            "iterations_used": iteration_counts + 1,
        }
