from anneal import run_abc
import torch
import random
import os
import json
from torch_geometric.data import Batch
import re
import subprocess
from model import PowerPredictor
from aig_encoder import load_aig_as_graph
from recipe_loader import load_recipes


# 🔥 same ops as training / SA
OPS = [
    "balance",
    "rewrite",
    "rewrite -z",
    "refactor",
    "refactor -z",
    "resub",
    "resub -z"
]


class PowerPredictorInference:
    def __init__(self, model_path, script_dir, norm_path, device="cpu"):
        self.device = torch.device(device)
        self.norm_path = norm_path

        # ---- vocab ----
        self.recipe_dict, self.vocab = load_recipes(script_dir)
        self.vocab_size = len(self.vocab)
        self.op_to_idx = self.vocab

        # ---- load norms ----
        if os.path.exists(norm_path):
            self.norms = torch.load(norm_path)
        else:
            self.norms = {}

        # ---- model ----
        self.model = PowerPredictor(self.vocab_size).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    # -------------------------
    # Encode recipe
    # -------------------------
    def encode_recipe(self, recipe_ops):
        return [self.op_to_idx[op] for op in recipe_ops]

    # -------------------------
    # Generate random recipe
    # -------------------------
    def random_recipe(self, length=20):
        return [random.choice(OPS) for _ in range(length)]

    # -------------------------
    # 🔥 NEW: Initialize unseen design
    # -------------------------
    def initialize_design(self, aig_path, design_name, K=15):
        """
        Runs K random recipes via ABC to estimate baseline
        """

        print(f"\n[Init] Computing baseline for unseen design: {design_name}")

        # from anneal_search import run_abc  # reuse your ABC runner

        powers = []

        for i in range(K):
            recipe = self.random_recipe()
            power = run_abc(aig_path, recipe)

            if power != float("inf"):
                powers.append(power)

            print(f"  sample {i:02d}: {power:.4f}")

        if len(powers) == 0:
            raise RuntimeError("All ABC runs failed. Cannot initialize.")

        # 🔥 same as training
        baseline = float(torch.tensor(powers).quantile(0.75))

        print(f"[Init] Baseline (75th percentile): {baseline:.4f}")

        # save
        self.norms[design_name] = {
            "baseline": baseline
        }

        torch.save(self.norms, self.norm_path)
        print("[Init] Saved to norms file")

    # -------------------------
    # Predict
    # -------------------------
    def predict(self, aig_path, recipe_ops, design_name):

        # 🔥 auto-init if unseen
        if design_name not in self.norms:
            self.initialize_design(aig_path, design_name)

        # --- graph ---
        graph = load_aig_as_graph(aig_path, cache_dir="cache/")
        graph = Batch.from_data_list([graph]).to(self.device)

        # --- recipe ---
        recipe_seq = self.encode_recipe(recipe_ops)
        recipe = torch.tensor([recipe_seq], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(recipe_seq)]).to(self.device)

        # --- baseline ---
        baseline_val = self.norms[design_name]["baseline"]
        baseline = torch.tensor([[baseline_val]], dtype=torch.float).to(self.device)

        # --- predict ---
        with torch.no_grad():
            delta = self.model(graph, recipe, lengths, baseline)

        return delta.item() + baseline_val

# =========================================================
# 🔥 DRIVER CODE (simple, no CLI)
# =========================================================

if __name__ == "__main__":

    # -------------------------
    # Init predictor
    # -------------------------
    predictor = PowerPredictorInference(
        model_path="checkpoints/best.pt",
        script_dir="data/scripts",
        norm_path="checkpoints/design_norms.pt",
        device="cuda"
    )

    # -------------------------
    # Inputs (EDIT HERE ONLY)
    # -------------------------
    design_name = "i2c"
    aig = "./data/designs/i2c.aig"

    recipe = [
        "refactor -z","balance","refactor -z","balance","rewrite",
        "rewrite -z","resub","resub","rewrite","resub -z",
        "refactor -z","rewrite","resub -z","refactor","refactor -z",
        "resub -z","balance","refactor -z","resub","balance"
    ]

    # -------------------------
    # Predict (auto-init if needed)
    # -------------------------
    power = predictor.predict(
        aig_path=aig,
        recipe_ops=recipe,
        design_name=design_name
    )
    real_power = run_abc(aig, recipe)
    print("\nPredicted Power:", power)
    print("\nReal Power:", real_power)