from anneal import run_abc
import torch
import os
from torch_geometric.data import Batch
from model import PowerPredictor
from aig_encoder import load_aig_as_graph
from recipe_loader import load_recipes
from predictor import PowerPredictorInference

# =========================================================
# DRIVER
# =========================================================
if __name__ == "__main__":

    predictor = PowerPredictorInference(
        model_path="checkpoints/best.pt",
        script_dir="data/scripts",
        device="cuda"
    )

    design_name = "square"
    aig = "./unseenDesigns/square.aig"

    # recipe = [
    #     "refactor -z","balance","refactor -z","balance","rewrite"
    # ]
    recipe = [
        "refactor -z","balance","refactor -z","balance","rewrite",
        "rewrite -z","resub","resub","rewrite","resub -z",
        "refactor -z","rewrite","resub -z","refactor","refactor -z",
        "resub -z","balance","refactor -z","resub","balance"
    ]
    predictor.adapt(aig, design_name, K=20)
    power      = predictor.predict(aig_path=aig, recipe_ops=recipe, design_name=design_name)
    real_power = run_abc(aig, recipe)

    print("\nPredicted Power:", power)
    print("Real Power:     ", real_power)