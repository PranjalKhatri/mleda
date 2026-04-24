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
import random


# -------------------------
# CSV loader
# -------------------------
def load_design_csv(power_csv, stats_csv):
    power_df = pd.read_csv(power_csv)
    stats_df = pd.read_csv(stats_csv)
    df = pd.merge(power_df, stats_df, on="sid")
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
            "df": df
        })

    return designs


# -------------------------
# Train
# -------------------------
def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- paths ----
    script_dir = "data/scripts"
    design_dir = "data/designs"
    power_dir = "data/power"
    stats_dir = "data/stats"
    cache_dir = "cache/"

    # ---- load recipes ----
    recipe_dict, vocab = load_recipes(script_dir)
    vocab_size = len(vocab)

    print(f"Vocab size: {vocab_size}")

    # ---- load designs ----
    designs = build_designs(design_dir, power_dir, stats_dir, cache_dir)
    print(f"Loaded {len(designs)} designs.")
    # ---- dataset ----
    dataset = PowerDataset(designs, recipe_dict)

    print(f"Total samples: {len(dataset)}")

    # ---- split ----
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ---- model ----
    model = PowerPredictor(vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ---- training loop ----
    for epoch in range(20):
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
                loss = criterion(pred, target)

                val_loss += loss.item()

        print(f"Epoch {epoch:02d} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ---- save model ----
    torch.save(model.state_dict(), "power_predictor.pt")
    print("Model saved.")


if __name__ == "__main__":
    train()