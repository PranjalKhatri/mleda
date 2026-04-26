# from anneal import run_abc
import torch
from torch_geometric.data import Batch

from model import PowerPredictor
from aig_encoder import load_aig_as_graph
from recipe_loader import load_recipes
import os
import random
import subprocess
import re
# -------------------------
# Allowed ops
# -------------------------
OPS = [
    "balance",
    "rewrite",
    "rewrite -z",
    "refactor",
    "refactor -z",
    "resub",
    "resub -z"
]

# -------------------------
# ABC evaluation (REAL)
# -------------------------
def run_abc(aig_path, recipe, lib_path="nangate45.lib"):
    """
    Runs ABC with mapping + power estimation using Nangate45
    Returns: float power
    """

    # build ABC script
    recipe_str = "; ".join(recipe)

    abc_cmd = (
        f"read_lib {lib_path}; "
        f"read {aig_path}; "
        f"{recipe_str}; "
        f"map; "
        f"print_stats -p;"
    )

    try:
        result = subprocess.run(
            ["abc", "-c", abc_cmd],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout

        # -------------------------
        # 🔥 Extract power via regex
        # -------------------------
        # Example line:
        # i/o = 177/128 lat = 0 and = 1169 lev = 14 power = 926.44

        match = re.search(r"power\s*=\s*([0-9]*\.?[0-9]+)", output)

        if not match:
            raise ValueError("Power not found in ABC output")

        power = float(match.group(1))
        return power

    except subprocess.CalledProcessError as e:
        print("ABC failed!")
        print(e.stderr)
        return float("inf")


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


# class PowerPredictorInference:
#     def __init__(self, model_path, script_dir, norm_path, device="cpu"):
#         self.device = torch.device(device)

#         # ---- load vocab ----
#         self.recipe_dict, self.vocab = load_recipes(script_dir)
#         self.vocab_size = len(self.vocab)
#         self.op_to_idx = self.vocab

#         # ---- load baseline info only ----
#         self.norms = torch.load(norm_path)

#         # ---- load model ----
#         self.model = PowerPredictor(self.vocab_size).to(self.device)
#         ckpt = torch.load(model_path, map_location=self.device)
#         self.model.load_state_dict(ckpt["model"])
#         self.model.eval()

#     # -------------------------
#     # Encode recipe
#     # -------------------------
#     def encode_recipe(self, recipe_ops):
#         seq = []
#         for op in recipe_ops:
#             if op not in self.op_to_idx:
#                 raise ValueError(f"Unknown op: {op}")
#             seq.append(self.op_to_idx[op])
#         return seq

#     # -------------------------
#     # Predict power
#     # -------------------------
#     def predict(self, aig_path, recipe_ops, design_name):

#         # --- graph ---
#         graph = load_aig_as_graph(aig_path, cache_dir="cache/")
#         graph = Batch.from_data_list([graph]).to(self.device)

#         # --- recipe ---
#         recipe_seq = self.encode_recipe(recipe_ops)
#         recipe = torch.tensor([recipe_seq], dtype=torch.long).to(self.device)
#         lengths = torch.tensor([len(recipe_seq)]).to(self.device)

#         # --- baseline (from training) ---
#         if design_name not in self.norms:
#             raise ValueError(f"Design {design_name} not found in norms file")

#         baseline_val = self.norms[design_name]["baseline"]
#         baseline = torch.tensor([[baseline_val]], dtype=torch.float).to(self.device)

#         # --- predict delta ---
#         with torch.no_grad():
#             delta = self.model(graph, recipe, lengths, baseline)

#         pred_power = delta.item() + baseline_val
#         return pred_power