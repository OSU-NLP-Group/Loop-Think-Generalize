import re
import json
import torch
import torch.nn as nn
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from torch.utils.data import Dataset


class CompositionDataset(Dataset):
    def __init__(self, file_path, vocab):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
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

        if target_tokens and target_tokens[-1] == "</a>":
            target_tokens = target_tokens[:-1]

        input_ids = torch.tensor(self.tokens_to_ids(input_tokens), dtype=torch.long)
        target_ids = torch.tensor(self.tokens_to_ids(target_tokens), dtype=torch.long)

        return input_ids, target_ids


class CompositionTestDataset(Dataset):
    def __init__(self, file_path, vocab, split=None):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        if split:
            self.data = [d for d in self.data if d.get("type") == split]
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

        if target_tokens and target_tokens[-1] == "</a>":
            target_tokens = target_tokens[:-1]

        input_ids = torch.tensor(self.tokens_to_ids(input_tokens), dtype=torch.long)
        target_ids = torch.tensor(self.tokens_to_ids(target_tokens), dtype=torch.long)

        return input_ids, target_ids, test_type


def custom_collate(batch):
    input_ids_list, target_ids_list = zip(*batch)
    pad_idx = 0

    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_idx)
    target_tokens = torch.tensor([x[-1].item() for x in target_ids_list], dtype=torch.long)
    attention_mask = (padded_input_ids != pad_idx).long()
    input_lengths = torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.long)

    return padded_input_ids, target_tokens, attention_mask, input_lengths


def custom_collate_test(batch):
    input_ids_list, target_ids_list, test_type = zip(*batch)
    pad_idx = 0

    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_idx)
    target_tokens = torch.tensor([x[-1].item() for x in target_ids_list], dtype=torch.long)
    attention_mask = (padded_input_ids != pad_idx).long()
    input_lengths = torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.long)

    return padded_input_ids, target_tokens, attention_mask, input_lengths, test_type


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab_list = json.load(f)
    if "<pad>" not in vocab_list:
        vocab_list = ["<pad>"] + vocab_list
    vocab = {token: idx for idx, token in enumerate(vocab_list)}
    return vocab, len(vocab)


def evaluate_model_test(model, dataloader, device, pad_idx=0):
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
            logits = outputs.logits

            predicted_tokens = torch.argmax(logits[:, -1, :], dim=-1)

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


class RecurrentGPT2Block(nn.Module):
    def __init__(self, config, num_iterations):
        super().__init__()
        self.config = config
        self.num_iterations = num_iterations
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.n_positions, config.n_embd)

        embd_pdrop = getattr(config, "embd_pdrop", 0.1)
        self.dropout = nn.Dropout(embd_pdrop)

        layer_norm_epsilon = getattr(config, "layer_norm_epsilon", 1e-5)
        self.ln_f = nn.LayerNorm(config.n_embd, eps=layer_norm_epsilon)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, attention_mask=None):

        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + pos_embeds
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        for _ in range(self.num_iterations):
            for block in self.blocks:
                hidden_states = block(hidden_states, attention_mask=attention_mask)[0]

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return type("Output", (object,), {"logits": logits})
