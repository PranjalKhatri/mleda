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

    df = pd.merge(power_df, stats_df, on="sid")

    # normalize column names (still useful for future)
    df.rename(columns={
        'pi': 'PI',
        'po': 'PO',
        'nd': 'AND',
        'lev': 'Level'
    }, inplace=True)

    return df


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
            "name": name
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

    # -------------------------
    # 🔥 SAVE BASELINE ONLY (stats removed)
    # -------------------------
    norm_dict = {}

    for design in designs:
        df = design['df']
        name = design['name']

        baseline_power = df['Power'].quantile(0.75)

        norm_dict[name] = {
            "baseline": float(baseline_power)
        }

    torch.save(norm_dict, os.path.join(ckpt_dir, "design_norms.pt"))
    print("Saved design-wise baseline.")

    # ---- dataset ----
    dataset = PowerDataset(designs, recipe_dict)
    print(f"Total samples: {len(dataset)}")

    # ---- split ----
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, collate_fn=collate_fn)

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

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    for epoch in range(start_epoch, 40):
        model.train()
        total_loss = 0

        for graph, recipe, lengths, baseline, target in train_loader:
            graph = graph.to(device)
            recipe = recipe.to(device)
            lengths = lengths.to(device)
            baseline = baseline.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            pred = model(graph, recipe, lengths, baseline)
            loss = criterion(pred, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---- validation ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for graph, recipe, lengths, baseline, target in val_loader:
                graph = graph.to(device)
                recipe = recipe.to(device)
                lengths = lengths.to(device)
                baseline = baseline.to(device)
                target = target.to(device)

                pred = model(graph, recipe, lengths, baseline)
                val_loss += criterion(pred, target).item()

        print(f"Epoch {epoch:02d} | Train: {total_loss:.4f} | Val: {val_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, best_val, f"{ckpt_dir}/last.pt")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, best_val, f"{ckpt_dir}/best.pt")
            print("Saved BEST checkpoint")

    # ---- test ----
    print("\nRunning test set...")
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for graph, recipe, lengths, baseline, target in test_loader:
            graph = graph.to(device)
            recipe = recipe.to(device)
            lengths = lengths.to(device)
            baseline = baseline.to(device)
            target = target.to(device)

            pred = model(graph, recipe, lengths, baseline)
            test_loss += criterion(pred, target).item()

    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    train(resume=False)