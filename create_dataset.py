import json
import os
import random

random.seed(42)
import numpy as np

np.random.seed(42)
import shutil

base_path = 'data/multi_hop_temp/'
output_dir = 'data/multi_hop/'
maxhop = 40


def combine_facts(*fact_lists):
    combined = []
    types = ['atomic']
    for i in range(len(fact_lists)):
        types.append(str(i) + "hop_train")
    for facts, tp in zip(fact_lists, types):
        for fact in facts:
            fact["type"] = tp
            combined.append(fact)
    return combined


def save_combined_facts(combined_facts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_facts, f)
    print(f"Saved combined facts to {output_file}")


def add_type_label(facts, fact_type):
    for fact in facts:
        fact['type'] = fact_type
    return facts


def random_sample_with_type(facts, sample_size, fact_type):
    sampled_facts = random.sample(facts, min(sample_size, len(facts)))
    return add_type_label(sampled_facts, fact_type)


with open(os.path.join(base_path, 'atomic_facts.json'), 'r', encoding='utf-8') as f:
    atomic_facts = json.load(f)

train_files = {}
test_files = {}
for i in range(2, maxhop + 1):
    train_files[f"{i}hop_train"] = os.path.join(base_path, f"{i}hop_train.json")
    test_files[f"{i}hop_test"] = os.path.join(base_path, f"{i}hop_test.json")

os.makedirs(output_dir, exist_ok=True)

shutil.copy(os.path.join(base_path, 'vocab.json'), os.path.join(output_dir, 'vocab.json'))
print(f"Copied vocab.json from {base_path} to {output_dir}")

train_data = {hop: json.load(open(path, 'r', encoding='utf-8')) for hop, path in train_files.items()}
test_data = {hop: json.load(open(path, 'r', encoding='utf-8')) for hop, path in test_files.items()}

print("Creating train sets...")
for i in range(2, maxhop + 1):
    train_set = combine_facts(atomic_facts, *[train_data[f"{j}hop_train"] for j in range(2, i + 1)])
    if i == 40:
        save_combined_facts(train_set, os.path.join(output_dir, f'train.json'))

print("Creating extended test set...")
extended_test_set = []
for i in range(2, maxhop + 1):
    extended_test_set.extend(add_type_label(test_data[f"{i}hop_test"], f"{i}hop_test"))

print("Creating small test set with 50 instances per type...")
small_test_set = []
for i in range(2, maxhop + 1):
    small_test_set.extend(random_sample_with_type(test_data[f"{i}hop_test"], 50, f"{i}hop_test"))

small_test_path = os.path.join(output_dir, 'test_small.json')
save_combined_facts(small_test_set, small_test_path)

extended_test_set.extend(random_sample_with_type(atomic_facts, 2000, "atomic_facts"))
for i in range(2, maxhop + 1):
    extended_test_set.extend(random_sample_with_type(train_data[f"{i}hop_train"], 2000, f"{i}hop_train"))

test_extended_path = os.path.join(output_dir, 'test.json')
save_combined_facts(extended_test_set, test_extended_path)

shutil.copy(test_extended_path, os.path.join(output_dir, 'valid.json'))
print(f"Created valid.json by copying test.json to {output_dir}")

print("All train, test, and valid sets created and saved.")
