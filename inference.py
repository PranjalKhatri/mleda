import torch
from predictor import PowerPredictorInference

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
# Inputs
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
# Predict
# -------------------------
power = predictor.predict(
    aig_path=aig,
    recipe_ops=recipe,
    design_name=design_name
)

print("Predicted Power:", power)