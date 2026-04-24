import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset_builder import PowerDataset
from collate import collate_fn
from model import PowerPredictor
from recipe_loader import load_recipes
from aig_encoder import load_aig_as_graph

import pandas as pd
import os


# -------------------------
# CSV loader
# -------------------------
def load_design_csv(power_csv, stats_csv):
    power_df = pd.read_csv(power_csv)
    stats_df = pd.read_csv(stats_csv)
    return pd.merge(power_df, stats_df, on="sid")


# -------------------------
# Build designs
# -------------------------
def build_designs(design_dir, power_dir, stats_dir, cache_dir):
    designs = []

    for file in os.listdir(design_dir):
        if not file.endswith(".aig"):
            continue

        name = file.replace(".aig", "")

        aig_path = os.path.join(design_dir, file)
        power_csv = os.path.join(power_dir, f"{name}_power.csv")
        stats_csv = os.path.join(stats_dir, f"{name}_stats.csv")

        if not os.path.exists(power_csv) or not os.path.exists(stats_csv):
            continue

        print(f"Loading design: {name}")

        graph = load_aig_as_graph(aig_path, cache_dir=cache_dir)
        df = load_design_csv(power_csv, stats_csv)

        designs.append({
            "graph": graph,
            "df": df,
            "name": name   # 🔥 ADD THIS
        })

    return designs


# -------------------------
# Save checkpoint
# -------------------------
def save_checkpoint(model, optimizer, epoch, best_val, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val
    }, path)


# -------------------------
# Load checkpoint
# -------------------------
def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["best_val"]


# -------------------------
# Train
# -------------------------
def train(resume=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- paths ----
    script_dir = "data/scripts"
    design_dir = "data/designs"
    power_dir = "data/power"
    stats_dir = "data/stats"
    cache_dir = "cache/"
    ckpt_dir = "checkpoints/"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- load recipes ----
    recipe_dict, vocab = load_recipes(script_dir)
    vocab_size = len(vocab)

    print(f"Vocab size: {vocab_size}")

    # ---- load designs ----
    designs = build_designs(design_dir, power_dir, stats_dir, cache_dir)
    print(f"Loaded {len(designs)} designs.")

    norm_dict = {}

    for design in designs:
        df = design['df']

        cols = ['BUFF','NOT','AND','PI','PO','LP']

        mean = df[cols].mean()
        std = df[cols].std() + 1e-6

        baseline_power = df['power'].quantile(0.75)

        # store using design name (IMPORTANT)
        # you need to add name in build_designs()
        name = design.get("name", None)
        if name is None:
            raise ValueError("Design name missing. Add it in build_designs().")

        norm_dict[name] = {
            "mean": mean.values.tolist(),
            "std": std.values.tolist(),
            "baseline": float(baseline_power)
        }

    # save once
    torch.save(norm_dict, os.path.join(ckpt_dir, "design_norms.pt"))
    print("Saved design-wise normalization.")

    dataset = PowerDataset(designs, recipe_dict)
    print(f"Total samples: {len(dataset)}")

    # ---- split ----
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # ---- model ----
    model = PowerPredictor(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_val = float("inf")

    # ---- resume ----
    if resume and os.path.exists(f"{ckpt_dir}/last.pt"):
        print("Resuming training...")
        start_epoch, best_val = load_checkpoint(
            model, optimizer, f"{ckpt_dir}/last.pt", device
        )

    # ---- training loop ----
    for epoch in range(start_epoch, 40):
        model.train()
        total_loss = 0

        for graph, recipe, lengths, stats, baseline, target in train_loader:
            graph = graph.to(device)
            recipe = recipe.to(device)
            lengths = lengths.to(device)
            stats = stats.to(device)
            baseline = baseline.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            pred = model(graph, recipe, lengths, stats, baseline)
            loss = criterion(pred, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---- validation ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for graph, recipe, lengths, stats, baseline, target in val_loader:
                graph = graph.to(device)
                recipe = recipe.to(device)
                lengths = lengths.to(device)
                stats = stats.to(device)
                baseline = baseline.to(device)
                target = target.to(device)

                pred = model(graph, recipe, lengths, stats, baseline)
                val_loss += criterion(pred, target).item()

        print(f"Epoch {epoch:02d} | Train: {total_loss:.4f} | Val: {val_loss:.4f}")

        # ---- save last ----
        save_checkpoint(model, optimizer, epoch, best_val, f"{ckpt_dir}/last.pt")

        # ---- save best ----
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, best_val, f"{ckpt_dir}/best.pt")
            print("Saved BEST checkpoint")

    # ---- test ----
    print("\nRunning test set...")
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for graph, recipe, lengths, stats, baseline, target in test_loader:
            graph = graph.to(device)
            recipe = recipe.to(device)
            lengths = lengths.to(device)
            stats = stats.to(device)
            baseline = baseline.to(device)
            target = target.to(device)

            pred = model(graph, recipe, lengths, stats, baseline)
            test_loss += criterion(pred, target).item()

    print(f"Test Loss: {test_loss:.4f}")


# -------------------------
# Inference
# -------------------------
def predict(model_path, aig_path, recipe_seq, stats, baseline, vocab, device="cpu"):

    device = torch.device(device)

    # load model
    model = PowerPredictor(len(vocab)).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # graph
    graph = load_aig_as_graph(aig_path, cache_dir="cache/")
    from torch_geometric.data import Batch
    graph = Batch.from_data_list([graph]).to(device)

    # recipe
    recipe = torch.tensor([recipe_seq], dtype=torch.long).to(device)
    lengths = torch.tensor([len(recipe_seq)]).to(device)

    stats = torch.tensor([stats], dtype=torch.float).to(device)
    baseline = torch.tensor([[baseline]], dtype=torch.float).to(device)

    with torch.no_grad():
        delta = model(graph, recipe, lengths, stats, baseline)

    pred_power = delta.item() + baseline.item()

    return pred_power


if __name__ == "__main__":
    train(resume=False)