import os


def load_recipes(script_dir):
    vocab = {}
    vocab_idx = 1  # reserve 0 for padding
    recipes = {}

    script_files = sorted([
        f for f in os.listdir(script_dir)
        if f.startswith("script") and f.endswith(".txt")
    ])

    for file in script_files:
        sid = int(file.replace("script", "").replace(".txt", ""))
        path = os.path.join(script_dir, file)

        with open(path, "r") as f:
            ops = [line.strip() for line in f if line.strip()]

        seq = []
        for op in ops:
            if op not in vocab:
                vocab[op] = vocab_idx
                vocab_idx += 1
            seq.append(vocab[op])

        recipes[sid] = seq

    return recipes, vocab