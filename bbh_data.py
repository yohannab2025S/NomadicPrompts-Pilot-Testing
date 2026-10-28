"""Load BIG-Bench Hard tasks (HF dataset 'lukaemon/bbh', fields input/target) and make the fixed-seed Part A/B split."""
import json, os, random
import config

def load_task(task):
    from datasets import load_dataset
    items = []
    for cfg in config.TASKS[task]["hf_configs"]:
        ds = load_dataset("lukaemon/bbh", cfg, split="test")
        for i, row in enumerate(ds):
            items.append(dict(id=f"{cfg}_{i:04d}", input=row["input"], target=row["target"]))
    return items

def get_split(task):
    path = f"data/{task}_split.json"
    if os.path.exists(path):
        return json.load(open(path))
    items = load_task(task)
    rng = random.Random(config.SEED)
    rng.shuffle(items)
    na, nb = config.TASKS[task]["n_a"], config.TASKS[task]["n_b"]
    assert na + nb <= len(items), f"{task}: only {len(items)} items"
    split = dict(part_a=items[:na], part_b=items[na:na + nb], seed=config.SEED, n_total=len(items))
    json.dump(split, open(path, "w"), indent=1)
    return split

def build_prompt(task, instruction, item):
    head = (instruction.strip() + "\n\n") if instruction else ""
    return f"{head}{item['input']}\n\n{config.FORMAT_LINES[task]}"
