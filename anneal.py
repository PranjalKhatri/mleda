import random
import math
import subprocess
import tempfile
import os
import re

from predictor import PowerPredictorInference


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
# Initial recipe (c2rs style)
# -------------------------
def get_initial_recipe(length=20):
    base = [
        "refactor -z", "balance",
        "rewrite", "rewrite -z",
        "resub", "resub -z",
        "balance"
    ]
    return (base * (length // len(base) + 1))[:length]


# -------------------------
# Mutation operators
# -------------------------
def mutate(recipe):
    r = recipe.copy()
    op = random.choice(["swap", "replace", "insert", "delete"])

    if op == "swap" and len(r) >= 2:
        i, j = random.sample(range(len(r)), 2)
        r[i], r[j] = r[j], r[i]

    elif op == "replace":
        i = random.randrange(len(r))
        r[i] = random.choice(OPS)

    elif op == "insert":
        i = random.randrange(len(r))
        r.insert(i, random.choice(OPS))

    elif op == "delete" and len(r) > 5:
        i = random.randrange(len(r))
        r.pop(i)

    return r


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

def extract_power(aig_path):
    # placeholder (replace this!)
    return random.uniform(500, 1500)


# -------------------------
# Simulated Annealing
# -------------------------
def simulated_annealing(
    predictor,
    aig_path,
    design_name,
    max_iters=200,
    T_init=10.0,
    cooling=0.95
):

    current = get_initial_recipe()
    current_score = predictor.predict(
        aig_path,
        current,
        design_name
    )

    best = current
    best_score = current_score

    T = T_init

    print(f"Initial predicted power: {current_score:.4f}")

    for it in range(max_iters):

        candidate = mutate(current)

        pred = predictor.predict(
            aig_path,
            candidate,
            design_name
        )

        delta = pred - current_score

        # accept rule
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = candidate
            current_score = pred

        if current_score < best_score:
            best = current
            best_score = current_score

        T *= cooling

        if it % 10 == 0:
            print(f"[{it}] Best pred: {best_score:.4f}")

    return best, best_score


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    predictor = PowerPredictorInference(
        model_path="checkpoints/best.pt",
        script_dir="data/scripts",
        norm_path="checkpoints/design_norms.pt",
        device="cuda"
    )

    aig = "./data/designs/i2c.aig"
    design_name = "i2c"

    best_recipe, pred_power = simulated_annealing(
        predictor,
        aig,
        design_name,
        max_iters=200
    )

    print("\n🔥 Best Recipe Found:")
    print(best_recipe)
    print("Predicted Power:", pred_power)

    # final ground truth
    real_power = run_abc(aig, best_recipe)

    print("\n⚡ Real Power:", real_power)