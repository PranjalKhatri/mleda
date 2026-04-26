import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_builder import PowerDataset
from collate import collate_fn
from model import PowerPredictor
from recipe_loader import load_recipes
from aig_encoder import load_aig_as_graph

import pandas as pd
import os


# -------------------------
# CSV loader (same as train)
# -------------------------
def load_design_csv(power_csv, stats_csv):
    power_df = pd.read_csv(power_csv)
    stats_df = pd.read_csv(stats_csv)

    df = pd.merge(power_df, stats_df, on="sid")

    df.rename(columns={
        'pi': 'PI',
        'po': 'PO',
        'nd': 'AND',
        'lev': 'Level'
    }, inplace=True)

    return df


# -------------------------
# Build ONLY new designs
# -------------------------
def build_designs(design_dir, power_dir, stats_dir, cache_dir, design_list):
    designs = []

    for name in design_list:
        aig_path = os.path.join(design_dir, f"{name}.aig")
        power_csv = os.path.join(power_dir, f"{name}_power.csv")
        stats_csv = os.path.join(stats_dir, f"{name}_stats.csv")

        if not (os.path.exists(aig_path) and
                os.path.exists(power_csv) and
                os.path.exists(stats_csv)):
            print(f"Skipping {name}, missing files")
            continue

        print(f"[FineTune] Loading design: {name}")

        graph = load_aig_as_graph(aig_path, cache_dir=cache_dir)
        df = load_design_csv(power_csv, stats_csv)

        designs.append({
            "graph": graph,
            "df": df,
            "name": name
        })

    return designs


# -------------------------
# Fine-tune
# -------------------------
def fine_tune():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- paths ----
    script_dir = "data/scripts"
    design_dir = "data/designs"
    power_dir = "data/power"
    stats_dir = "data/stats"
    cache_dir = "cache/"
    ckpt_path = "checkpoints/best.pt"

    # 🔥 specify new designs here
    new_designs = ["sqrt"]   # change this

    # ---- recipes ----
    recipe_dict, vocab = load_recipes(script_dir)
    vocab_size = len(vocab)

    # ---- dataset ----
    designs = build_designs(
        design_dir,
        power_dir,
        stats_dir,
        cache_dir,
        new_designs
    )

    dataset = PowerDataset(designs, recipe_dict)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    print(f"[FineTune] Samples: {len(dataset)}")

    # ---- model ----
    model = PowerPredictor(vocab_size).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    print("[FineTune] Loaded pretrained model")

    # 🔥 OPTIONAL: freeze graph encoder (recommended)
    for p in model.graph_encoder.parameters():
        p.requires_grad = False

    # ---- optimizer ----
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4   # smaller LR for fine-tuning
    )

    criterion = nn.MSELoss()

    # ---- training ----
    epochs = 5   # keep small

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for graph, recipe, lengths, baseline, target in loader:
            graph = graph.to(device)
            recipe = recipe.to(device)
            lengths = lengths.to(device)
            baseline = baseline.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # 🔥 NO STATS
            pred = model(graph, recipe, lengths, baseline)

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[FineTune] Epoch {epoch} | Loss: {total_loss:.4f}")

    # ---- save ----
    torch.save({
        "model": model.state_dict()
    }, "checkpoints/fine_tuned.pt")

    print("[FineTune] Saved to checkpoints/fine_tuned.pt")


if __name__ == "__main__":
    fine_tune()