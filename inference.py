from predictor import PowerPredictorInference
import torch
import os

predictor = PowerPredictorInference(
    model_path="checkpoints/best.pt",
    script_dir="data/scripts",
    device="cuda"
)

# -------------------------
# Load normalization info
# -------------------------
norms = torch.load("checkpoints/design_norms.pt")

design_name = "i2c"   # 🔥 must match filename
info = norms[design_name]

mean = info["mean"]
std = info["std"]
baseline = info["baseline"]   # 🔥 DO NOT hardcode anymore

# -------------------------
# Inputs
# -------------------------
aig = "./data/designs/i2c.aig"

recipe = [
    "refactor -z","balance","refactor -z","balance","rewrite",
    "rewrite -z","resub","resub","rewrite","resub -z",
    "refactor -z","rewrite","resub -z","refactor","refactor -z",
    "resub -z","balance","refactor -z","resub","balance"
]

# RAW stats (from ABC or wherever)
stats = [928,1030,915,177,128,15]

# -------------------------
# 🔥 Normalize stats (CRITICAL)
# -------------------------
stats_norm = [(s - m) / (st + 1e-6) for s, m, st in zip(stats, mean, std)]

# -------------------------
# Predict
# -------------------------
power = predictor.predict(aig, recipe, stats_norm, baseline)

print("Power:", power)