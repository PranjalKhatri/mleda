import torch
import numpy as np
from torch_geometric.data import Batch
from model import PowerPredictor
from aig_encoder import load_aig_as_graph
from recipe_loader import load_recipes
import os
import random
import subprocess
import re

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
# ABC evaluation
# -------------------------
def run_abc(aig_path, recipe, lib_path="nangate45.lib"):
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
            capture_output=True, text=True, check=True
        )
        match = re.search(r"power\s*=\s*([0-9]*\.?[0-9]+)", result.stdout)
        if not match:
            raise ValueError("Power not found in ABC output")
        return float(match.group(1))
    except subprocess.CalledProcessError as e:
        print("ABC failed!", e.stderr)
        return float("inf")


# -------------------------
# Inference class
# -------------------------
class PowerPredictorInference:
    """
    For trained designs  → predict() uses raw model output directly.

    For unseen designs   → call adapt(aig_path, design_name, K=20) first.
                           This runs K real ABC samples, compares model
                           predictions on those same recipes to real values,
                           and fits a linear correction  y = a*pred + b
                           (least-squares, takes ~seconds).
                           After adapt(), predict() applies the correction
                           automatically so predictions are anchored to the
                           real power range of the new design.
    """

    def __init__(self, model_path, script_dir, norm_path=None, device="cpu"):
        self.device    = torch.device(device)
        self.norm_path = norm_path

        self.recipe_dict, self.vocab = load_recipes(script_dir)
        self.vocab_size = len(self.vocab)
        self.op_to_idx  = self.vocab

        self.model = PowerPredictor(self.vocab_size).to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # design_name -> (a, b)  linear correction: real ≈ a*pred + b
        self._corrections: dict[str, tuple[float, float]] = {}

        # load previously saved corrections if available
        if norm_path and os.path.exists(norm_path):
            saved = torch.load(norm_path)
            if "corrections" in saved:
                self._corrections = saved["corrections"]

    # -------------------------
    def encode_recipe(self, recipe_ops):
        return [self.op_to_idx[op] for op in recipe_ops]

    def random_recipe(self, length=20):
        return [random.choice(OPS) for _ in range(length)]

    # -------------------------
    def _raw_predict(self, aig_path, recipe_ops):
        """Model forward pass, no correction applied."""
        graph = load_aig_as_graph(aig_path, cache_dir="cache/")
        graph = Batch.from_data_list([graph]).to(self.device)

        seq     = self.encode_recipe(recipe_ops)
        recipe  = torch.tensor([seq], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(seq)]).to(self.device)

        with torch.no_grad():
            pred = self.model(graph, recipe, lengths)

        return pred.item()

    # -------------------------
    def adapt(self, aig_path, design_name, K=20):
        """
        Test-time adaptation for an unseen design.

        Runs K diverse random recipes through ABC (real), gets the model's
        raw prediction for each, then fits a linear map:
            real_power ≈ a * model_pred + b
        via least squares.

        The model already learned *relative* recipe differences well —
        adaptation just anchors it to the correct power scale and offset
        for this new design. K=20 is usually enough; use K=30 if the
        design is very different from training designs.

        After calling adapt(), predict() automatically applies the
        correction for this design.
        """
        print(f"\n[Adapt] Fitting correction for unseen design: {design_name}")
        print(f"        Running {K} ABC samples ...")

        preds_raw = []
        reals     = []

        # use diverse seeds: spread across different op biases
        seed_recipes = self._diverse_seeds(K)

        for i, recipe in enumerate(seed_recipes):
            real = run_abc(aig_path, recipe)
            if real == float("inf"):
                continue

            raw = self._raw_predict(aig_path, recipe)
            preds_raw.append(raw)
            reals.append(real)
            print(f"  [{i:02d}] pred={raw:.2f}  real={real:.2f}")

        if len(reals) < 4:
            raise RuntimeError(
                f"Only {len(reals)} successful ABC runs — cannot fit correction."
            )

        # least-squares fit:  real = a * pred + b
        X = np.array(preds_raw).reshape(-1, 1)
        y = np.array(reals)
        # [X 1] @ [a, b]^T = y
        A = np.hstack([X, np.ones_like(X)])
        result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(result[0]), float(result[1])

        # sanity: if slope is negative the model's ordering is inverted,
        # fall back to just a mean shift (b only, a=1)
        if a <= 0:
            print(f"  [warn] Negative slope ({a:.3f}), falling back to mean shift.")
            a = 1.0
            b = float(np.mean(y) - np.mean(preds_raw))

        self._corrections[design_name] = (a, b)
        print(f"[Adapt] Correction for '{design_name}': "
              f"real ≈ {a:.4f} * pred + {b:.4f}")

        # save so we don't have to redo it next run
        if self.norm_path:
            existing = torch.load(self.norm_path) if os.path.exists(self.norm_path) else {}
            existing["corrections"] = self._corrections
            torch.save(existing, self.norm_path)
            print(f"[Adapt] Saved correction to {self.norm_path}")

        return a, b

    # -------------------------
    def predict(self, aig_path, recipe_ops, design_name=None):
        """
        Predict power for a recipe on an AIG.

        If design_name is given and adapt() has been called for it,
        applies the linear correction automatically.
        Call adapt() before predict() for unseen designs.
        """
        raw = self._raw_predict(aig_path, recipe_ops)

        if design_name and design_name in self._corrections:
            a, b = self._corrections[design_name]
            return a * raw + b

        return raw

    # -------------------------
    def _diverse_seeds(self, K):
        """
        Returns K recipes that cover different parts of the op space.
        First few are deterministic known-good recipes, rest are random.
        """
        seeds = [
            # standard c2rs-style
            ["balance","rewrite","rewrite -z","balance","rewrite -z",
             "balance","refactor","rewrite","rewrite -z","balance",
             "rewrite -z","balance","refactor","rewrite","rewrite -z",
             "balance","rewrite -z","balance","refactor","rewrite"],
            # resub heavy
            ["resub","resub -z","resub","balance","resub -z",
             "resub","rewrite","resub -z","resub","balance",
             "resub -z","resub","rewrite -z","resub","balance",
             "resub -z","resub","rewrite","resub -z","resub"],
            # refactor heavy
            ["refactor","refactor -z","balance","refactor","refactor -z",
             "balance","refactor","refactor -z","balance","refactor",
             "refactor -z","balance","refactor","refactor -z","balance",
             "refactor","refactor -z","balance","refactor","refactor -z"],
            # all balance
            ["balance"] * 20,
            # alternating rewrite/refactor
            ["rewrite","refactor","rewrite -z","refactor -z","balance",
             "rewrite","refactor","rewrite -z","refactor -z","balance",
             "rewrite","refactor","rewrite -z","refactor -z","balance",
             "rewrite","refactor","rewrite -z","refactor -z","balance"],
        ]

        # pad with random recipes up to K
        while len(seeds) < K:
            seeds.append(self.random_recipe())

        return seeds[:K]