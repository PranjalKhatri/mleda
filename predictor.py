import torch
from torch_geometric.data import Batch

from model import PowerPredictor
from aig_encoder import load_aig_as_graph
from recipe_loader import load_recipes


class PowerPredictorInference:
    def __init__(self, model_path, script_dir, norm_path, device="cpu"):
        self.device = torch.device(device)

        # ---- load vocab ----
        self.recipe_dict, self.vocab = load_recipes(script_dir)
        self.vocab_size = len(self.vocab)

        self.op_to_idx = self.vocab

        # ---- load normalization ----
        self.norms = torch.load(norm_path)

        # ---- load model ----
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
    # Normalize stats
    # -------------------------
    def normalize_stats(self, stats, design_name):
        info = self.norms[design_name]

        mean = info["mean"]
        std = info["std"]

        return [(s - m) / (st + 1e-6) for s, m, st in zip(stats, mean, std)]

    # -------------------------
    # Predict power
    # -------------------------
    def predict(self, aig_path, recipe_ops, raw_stats, design_name):

        # --- graph ---
        graph = load_aig_as_graph(aig_path, cache_dir="cache/")
        graph = Batch.from_data_list([graph]).to(self.device)

        # --- recipe ---
        recipe_seq = self.encode_recipe(recipe_ops)
        recipe = torch.tensor([recipe_seq], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(recipe_seq)]).to(self.device)

        # --- normalize stats ---
        stats_norm = self.normalize_stats(raw_stats, design_name)
        stats = torch.tensor([stats_norm], dtype=torch.float).to(self.device)

        # --- baseline (from training) ---
        baseline_val = self.norms[design_name]["baseline"]
        baseline = torch.tensor([[baseline_val]], dtype=torch.float).to(self.device)

        # --- predict delta ---
        with torch.no_grad():
            delta = self.model(graph, recipe, lengths, stats, baseline)

        pred_power = delta.item() + baseline_val
        return pred_power