import numpy as np
from tqdm.auto import tqdm
import os
import json
import random

np.random.seed(42)
random.seed(42)

NUM_ENTITY_IN = 200
NUM_RELATION = 10
max_hop = 40
data_dir = 'data/multi_hop_temp'


def build_dicts(entities):
    entity2ind = dict()
    ind2entity = []
    for i in range(len(entities)):
        entity = entities[i]
        if not (entity in ind2entity):
            ind2entity.append(entity)
            entity2ind[entity] = len(ind2entity) - 1
    return ind2entity, entity2ind


def choose(arr, ratio_or_count):
    if type(ratio_or_count) == float:
        num = round(ratio_or_count * len(arr))
    elif type(ratio_or_count) == int:
        num = ratio_or_count
    else:
        assert False
    if num >= len(arr):
        return arr
    rand_inds = np.random.choice(len(arr), num, replace=False).tolist()
    return [arr[i] for i in rand_inds]


def split(arr, ratio_or_count):
    if type(ratio_or_count) == float:
        num = round(ratio_or_count * len(arr))
    elif type(ratio_or_count) == int:
        num = ratio_or_count
    else:
        assert False
    train, test = [], []
    rand_inds = np.random.choice(len(arr), num, replace=False).tolist()
    for i in tqdm(range(len(arr))):
        if i in rand_inds:
            train.append(arr[i])
        else:
            test.append(arr[i])
    return [train, test]


def form_items(fact):
    input_text = "".join(fact[:-1])
    target_text = input_text + fact[-1] + "</a>"
    item = {
        "input_text": input_text,
        "target_text": target_text
    }

    return item


def process_facts(facts):
    processed_facts = []

    for fact in facts:
        item = form_items(fact)
        processed_facts.append(item)

    return processed_facts


def save_facts_to_file(facts, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(facts, f)


def build_atomic(num_entities, num_relations, out_degree=10, seed=42):
    rng = np.random.default_rng(seed)

    entities = [f"<e_{i}>" for i in range(num_entities)]

    relations = [f"<r_{i}>" for i in range(num_relations)]

    perms = [rng.permutation(num_entities) for _ in range(num_relations)]

    atomic_dict = {}

    atomic_facts = []

    if out_degree != num_relations:
        print(

            f"[Warning] out_degree ({out_degree}) != num_relations ({num_relations}). "

            "In this case, a relation may not achieve full 1..N target coverage "

            "because some heads will not include that relation."

        )

    for i in tqdm(range(num_entities)):

        selected_rows = rng.choice(num_relations, size=out_degree, replace=False).tolist()

        h = entities[i]

        for row_idx in selected_rows:
            t_idx = perms[row_idx][i]

            r = relations[row_idx]

            t = entities[t_idx]

            atomic_facts.append([h, r, t])

            atomic_dict.setdefault(h, []).append((r, t))

    return entities, relations, atomic_facts


entities, relations, atomic_facts = build_atomic(NUM_ENTITY_IN, NUM_RELATION)

os.makedirs(data_dir, exist_ok=True)

processed_atomic_facts = process_facts(atomic_facts)
save_facts_to_file(processed_atomic_facts, os.path.join(data_dir, 'atomic_facts.json'))

print(len(atomic_facts))

vocab = []
vocab = vocab + entities + relations
vocab = vocab + ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"]
assert len(vocab) == len(set(vocab))
print("vocab size:", len(vocab))
with open(os.path.join(data_dir, 'vocab.json'), "w", encoding='utf-8') as f:
    json.dump(vocab, f)


def build_nhop_facts_fast(atomic_facts, n, max_facts=15750):
    nhop_facts = []
    fact_set = set()

    atomic_dict = {}
    for h, r, t in atomic_facts:
        if h not in atomic_dict:
            atomic_dict[h] = []
        atomic_dict[h].append((r, t))

    with tqdm(total=max_facts, desc=f'Generating {n}-hop facts') as pbar:
        while len(nhop_facts) < max_facts:
            start_fact = random.choice(atomic_facts)
            h, r1, t1 = start_fact
            fact_chain = [h, r1]

            for _ in range(1, n):
                last_tail = t1
                if last_tail in atomic_dict:
                    r_next, t_next = random.choice(atomic_dict[last_tail])
                    fact_chain.append(r_next)
                    t1 = t_next
                else:
                    break

            fact_chain.append(t1)

            fact_tuple = tuple(fact_chain)
            if len(fact_chain) == n + 2 and fact_tuple not in fact_set:
                nhop_facts.append(fact_chain)
                fact_set.add(fact_tuple)
                pbar.update(1)

    return nhop_facts


for hop in range(2, max_hop + 1):
    print(f"Generating {hop}-hop facts...")
    nhop_facts = build_nhop_facts_fast(atomic_facts, hop)
    processed_nhop_facts = process_facts(nhop_facts)
    save_facts_to_file(processed_nhop_facts, f'{data_dir}/{hop}hop_facts.json')
    print(f"{hop}-hop facts generated and saved to '{data_dir}/{hop}hop_facts.json'")


def load_facts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        facts = json.load(f)
    return facts


def print_random_examples(facts, num_examples=5):
    print(f"Random {num_examples} examples:")
    for example in random.sample(facts, num_examples):
        print(example)
    print("\n")


datasets = {}
for i in range(2, max_hop + 1):
    datasets[str(i) + '-hop'] = data_dir + '/' + str(i) + 'hop_facts.json'

for hop, file_path in datasets.items():
    facts = load_facts_from_file(file_path)
    print(f"{len(facts)}--- {hop} Facts ---")
    print_random_examples(facts)


def split_facts(facts, train_size=15000, test_size=750):
    random.shuffle(facts)
    train_facts = facts[:train_size]
    test_facts = facts[train_size:train_size + test_size]
    return train_facts, test_facts


def save_split_facts(train_facts, test_facts, hop, output_dir=data_dir):
    train_file = os.path.join(output_dir, f'{hop}hop_train.json')
    test_file = os.path.join(output_dir, f'{hop}hop_test.json')

    with open(train_file, 'w', encoding='utf-8') as f_train:
        json.dump(train_facts, f_train)

    with open(test_file, 'w', encoding='utf-8') as f_test:
        json.dump(test_facts, f_test)

    print(f"Saved {hop}-hop train (45000) and test (750) to {output_dir}")


datasets = {}
for i in range(2, max_hop + 1):
    datasets[str(i)] = data_dir + '/' + str(i) + 'hop_facts.json'

for hop, file_path in datasets.items():
    print(f"Splitting {hop}-hop facts...")
    with open(file_path, 'r', encoding='utf-8') as f:
        facts = json.load(f)

    train_facts, test_facts = split_facts(facts)

    save_split_facts(train_facts, test_facts, hop)
