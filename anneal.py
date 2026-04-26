import random
import math
import subprocess
import tempfile
import os
import re
import json
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
        "balance","rewrite","rewrite -z","balance","rewrite -z","balance","refactor"
    ]
    return (base * (length // len(base) + 1))[:length]


# -------------------------
# Mutation operators
# -------------------------
def mutate(recipe):
    r = recipe.copy()
    
    # Pick 1 to 5 mutations, with higher probability for larger numbers
    num_mutations = random.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 4, 5], k=1)[0]

    for _ in range(num_mutations):
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

    # Enforce exact recipe size of 20
    while len(r) > 20:
        r.pop(random.randrange(len(r)))
    while len(r) < 20:
        r.insert(random.randrange(len(r) + 1), random.choice(OPS))

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

    design_dir = "./data/designs"
    out_dir = "anneal_results"
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(design_dir):
        if not file.endswith(".aig"):
            continue

        design_name = file.replace(".aig", "")
        aig = os.path.join(design_dir, file)

        print(f"\n========================================")
        print(f"Processing design: {design_name}")
        print(f"========================================")

        try:
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

            res_path = os.path.join(out_dir, f"{design_name}_results.json")
            update_results = True

            if os.path.exists(res_path):
                try:
                    with open(res_path, "r") as f:
                        existing_data = json.load(f)
                    if "real_power" in existing_data and real_power >= existing_data["real_power"]:
                        update_results = False
                        print(f"📉 Keeping existing result: {existing_data['real_power']} is better than or equal to {real_power}")
                except Exception as e:
                    print(f"⚠️ Could not read existing results for {design_name}: {e}")

            if update_results:
                with open(res_path, "w") as f:
                    json.dump({
                        "design": design_name,
                        "best_recipe": best_recipe,
                        "predicted_power": pred_power,
                        "real_power": real_power
                    }, f, indent=4)
                print(f"✅ Results saved to {res_path}")

        except ValueError as e:
            print(f"⚠️ Skipping {design_name}: {e}")