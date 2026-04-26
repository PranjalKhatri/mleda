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
    df.rename(columns={'pi': 'PI', 'po': 'PO', 'nd': 'AND', 'lev': 'Level'}, inplace=True)
    return df


# -------------------------
# Build designs
# -------------------------
def build_designs(design_dir, power_dir, stats_dir, cache_dir):
    designs = []

    for file in sorted(os.listdir(design_dir)):
        if not file.endswith(".aig"):
            continue

        name      = file.replace(".aig", "")
        aig_path  = os.path.join(design_dir, file)
        power_csv = os.path.join(power_dir, f"{name}_power.csv")
        stats_csv = os.path.join(stats_dir, f"{name}_stats.csv")

        if not os.path.exists(power_csv) or not os.path.exists(stats_csv):
            print(f"  [skip] missing CSVs for {name}")
            continue

        print(f"Loading design: {name}")
        graph = load_aig_as_graph(aig_path, cache_dir=cache_dir)
        df    = load_design_csv(power_csv, stats_csv)
        designs.append({"graph": graph, "df": df, "name": name})

    return designs


# -------------------------
# Checkpointing
# -------------------------
def save_checkpoint(model, optimizer, scheduler, epoch, best_val, path):
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch":     epoch,
        "best_val":  best_val,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_val"]


# -------------------------
# Train
# -------------------------
def train(resume=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = "data/scripts"
    design_dir = "data/designs"
    power_dir  = "data/power"
    stats_dir  = "data/stats"
    cache_dir  = "cache/"
    ckpt_dir   = "checkpoints/"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- recipes ----
    recipe_dict, vocab = load_recipes(script_dir)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    # ---- designs ----
    designs = build_designs(design_dir, power_dir, stats_dir, cache_dir)
    print(f"Loaded {len(designs)} designs.")

    # -------------------------
    # Per-design power normalisation stats
    # Used ONLY at inference to denormalise predictions
    # NOT fed into the model — avoids the shortcut
    # -------------------------
    norm_dict = {}
    for d in designs:
        powers = d['df']['Power']
        norm_dict[d['name']] = {
            "mean": float(powers.mean()),
            "std":  float(powers.std() + 1e-8),
        }
    torch.save(norm_dict, os.path.join(ckpt_dir, "design_norms.pt"))
    print("Saved design_norms.pt")

    # ---- dataset ----
    dataset = PowerDataset(designs, recipe_dict)
    print(f"Total samples: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size   = int(0.1 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=2, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=2, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=2, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # ---- model ----
    model     = PowerPredictor(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss()   # equivalent to HuberLoss(delta=1.0), works on older PyTorch

    start_epoch = 0
    best_val    = float("inf")

    if resume and os.path.exists(f"{ckpt_dir}/last.pt"):
        print("Resuming training...")
        start_epoch, best_val = load_checkpoint(
            model, optimizer, scheduler, f"{ckpt_dir}/last.pt", device
        )

    # early stopping
    patience    = 10
    no_improve  = 0
    min_delta   = 1e-4

    # =========================================================
    # TRAIN LOOP
    # =========================================================
    for epoch in range(start_epoch, 20):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for graph, recipe, lengths, target in train_loader:
            graph   = graph.to(device)
            recipe  = recipe.to(device)
            lengths = lengths.to(device)
            target  = target.to(device)       # (B, 1) raw power

            optimizer.zero_grad()
            pred = model(graph, recipe, lengths)   # (B, 1)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_train = total_loss / max(n_batches, 1)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        n_val    = 0
        pct_errs = []

        with torch.no_grad():
            for graph, recipe, lengths, target in val_loader:
                graph   = graph.to(device)
                recipe  = recipe.to(device)
                lengths = lengths.to(device)
                target  = target.to(device)

                pred     = model(graph, recipe, lengths)
                val_loss += criterion(pred, target).item()
                n_val    += 1

                pct = ((pred - target).abs() / (target.abs() + 1e-8)) * 100
                pct_errs.append(pct)

        avg_val  = val_loss / max(n_val, 1)
        mean_pct = torch.cat(pct_errs).mean().item()
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(avg_val)

        print(
            f"Epoch {epoch:03d} | "
            f"Train: {avg_train:.6f} | "
            f"Val: {avg_val:.6f} | "
            f"Val Err%: {mean_pct:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        save_checkpoint(model, optimizer, scheduler, epoch, best_val,
                        f"{ckpt_dir}/last.pt")

        if avg_val < best_val - min_delta:
            best_val   = avg_val
            no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val,
                            f"{ckpt_dir}/best.pt")
            print("  --> Saved BEST checkpoint")
        else:
            no_improve += 1
            print(f"  --> No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # ---- test ----
    print("\nRunning test set...")
    model.eval()
    test_loss = 0.0
    n_test    = 0
    test_pct  = []

    with torch.no_grad():
        for graph, recipe, lengths, target in test_loader:
            graph   = graph.to(device)
            recipe  = recipe.to(device)
            lengths = lengths.to(device)
            target  = target.to(device)

            pred      = model(graph, recipe, lengths)
            test_loss += criterion(pred, target).item()
            n_test    += 1

            pct = ((pred - target).abs() / (target.abs() + 1e-8)) * 100
            test_pct.append(pct)

    print(f"Test Huber Loss:      {test_loss / max(n_test, 1):.6f}")
    print(f"Test Mean Abs Err%:   {torch.cat(test_pct).mean().item():.2f}%")


if __name__ == "__main__":
    train(resume=True)