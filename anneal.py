import random
import math
import subprocess
import os
import re
import json
import torch
from predictor import PowerPredictorInference, run_abc


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

RECIPE_LEN = 20


# -------------------------
# Initial recipe
# -------------------------
def get_initial_recipe():
    base = [
        "balance", "rewrite", "rewrite -z", "balance",
        "rewrite -z", "balance", "refactor"
    ]
    return (base * (RECIPE_LEN // len(base) + 1))[:RECIPE_LEN]


# -------------------------
# Mutation
# -------------------------
def mutate(recipe, temperature, t_init):
    """
    Mutation strength scales with temperature:
    hot  (early) → up to 2 mutations, broad exploration
    cold (late)  → 1 mutation, fine-tuning
    """
    r    = recipe.copy()
    heat = min(temperature / t_init, 1.0)
    n    = 2 if (heat > 0.5 and random.random() < heat) else 1

    for _ in range(n):
        op = random.choice(["replace", "replace", "swap", "insert_delete"])

        if op == "swap" and len(r) >= 2:
            i, j   = random.sample(range(len(r)), 2)
            r[i], r[j] = r[j], r[i]

        elif op == "replace":
            i    = random.randrange(len(r))
            opts = [o for o in OPS if o != r[i]]
            r[i] = random.choice(opts)

        elif op == "insert_delete":
            r.insert(random.randrange(len(r) + 1), random.choice(OPS))
            r.pop(random.randrange(len(r)))

    # enforce exact length
    while len(r) > RECIPE_LEN:
        r.pop(random.randrange(len(r)))
    while len(r) < RECIPE_LEN:
        r.insert(random.randrange(len(r) + 1), random.choice(OPS))

    return r


# -------------------------
# Simulated Annealing
# -------------------------
def simulated_annealing(
    predictor,
    aig_path,
    design_name,
    max_iters=500,
    T_init=190.0,
    T_min=0.1,
    reheat_every=100,
    reheat_factor=2.0,
    max_reheats=3,
):
    # cooling rate so T reaches T_min exactly at max_iters
    cooling = math.exp(math.log(T_min / T_init) / max_iters)

    current       = get_initial_recipe()
    current_score = predictor.predict(aig_path, current, design_name)
    best          = current[:]
    best_score    = current_score

    T                = T_init
    iters_no_improve = 0
    reheats_done     = 0
    accepted         = 0

    print(f"  Initial predicted power : {current_score:.4f}")
    print(f"  T_init={T_init}  T_min={T_min}  "
          f"cooling={cooling:.5f}  iters={max_iters}")

    for it in range(1, max_iters + 1):

        candidate     = mutate(current, T, T_init)
        pred          = predictor.predict(aig_path, candidate, design_name)
        delta         = pred - current_score

        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current       = candidate
            current_score = pred
            accepted     += 1

        if current_score < best_score:
            best             = current[:]
            best_score       = current_score
            iters_no_improve = 0
        else:
            iters_no_improve += 1

        # reheating
        if (iters_no_improve >= reheat_every
                and reheats_done < max_reheats
                and T > T_min):
            T                = min(T * reheat_factor, T_init * 0.5)
            reheats_done    += 1
            iters_no_improve = 0
            print(f"  [reheat {reheats_done}] T → {T:.4f}")
        else:
            T = max(T * cooling, T_min)

        if it % 50 == 0:
            print(f"  [{it:4d}/{max_iters}] best={best_score:.4f}  "
                  f"curr={current_score:.4f}  T={T:.4f}  "
                  f"accept={accepted/it*100:.1f}%")

    print(f"  Done. Best predicted power: {best_score:.4f}")
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
    out_dir    = "anneal_results"
    os.makedirs(out_dir, exist_ok=True)

    # designs the model was trained on — no adaptation needed for these
    norms_path = "checkpoints/design_norms.pt"
    if os.path.exists(norms_path):
        norms = torch.load(norms_path)
        trained_designs = set(norms.get("corrections", {}).keys()) | (
            set(norms.keys()) - {"corrections"}
        )
    else:
        trained_designs = set()

    print(f"Trained designs ({len(trained_designs)}): {sorted(trained_designs)}")

    for file in sorted(os.listdir(design_dir)):
        if not file.endswith(".aig"):
            continue

        design_name = file.replace(".aig", "")
        aig         = os.path.join(design_dir, file)

        print(f"\n{'='*50}")
        print(f"Design: {design_name}")
        print(f"{'='*50}")

        try:
            # ---- adapt only for genuinely unseen designs ----
            if design_name in trained_designs:
                print(f"  Trained design — skipping adaptation.")
            elif design_name in predictor._corrections:
                print(f"  Correction already cached — skipping adaptation.")
            else:
                print(f"  Unseen design — running test-time adaptation ...")
                predictor.adapt(aig, design_name, K=20)

            best_recipe, pred_power = simulated_annealing(
                predictor, aig, design_name, max_iters=500,
            )

            print(f"\n  Best Recipe : {best_recipe}")
            print(f"  Predicted   : {pred_power:.4f}")

            real_power = run_abc(aig, best_recipe)
            print(f"  Real Power  : {real_power:.4f}")

            # ---- save if better than existing ----
            res_path = os.path.join(out_dir, f"{design_name}_results.json")
            save     = True

            if os.path.exists(res_path):
                try:
                    with open(res_path) as f:
                        existing = json.load(f)
                    if existing.get("real_power", float("inf")) <= real_power:
                        save = False
                        print(f"  Keeping existing: "
                              f"{existing['real_power']:.4f} <= {real_power:.4f}")
                except Exception as e:
                    print(f"  Could not read existing results: {e}")

            if save:
                with open(res_path, "w") as f:
                    json.dump({
                        "design":          design_name,
                        "best_recipe":     best_recipe,
                        "predicted_power": pred_power,
                        "real_power":      real_power,
                    }, f, indent=4)
                print(f"  Saved → {res_path}")

        except Exception as e:
            print(f"  Skipping {design_name}: {e}")