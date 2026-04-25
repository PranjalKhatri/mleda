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

        # ---- load baseline info only ----
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
        seq = []
        for op in recipe_ops:
            if op not in self.op_to_idx:
                raise ValueError(f"Unknown op: {op}")
            seq.append(self.op_to_idx[op])
        return seq

    # -------------------------
    # Predict power
    # -------------------------
    def predict(self, aig_path, recipe_ops, design_name):

        # --- graph ---
        graph = load_aig_as_graph(aig_path, cache_dir="cache/")
        graph = Batch.from_data_list([graph]).to(self.device)

        # --- recipe ---
        recipe_seq = self.encode_recipe(recipe_ops)
        recipe = torch.tensor([recipe_seq], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(recipe_seq)]).to(self.device)

        # --- baseline (from training) ---
        if design_name not in self.norms:
            raise ValueError(f"Design {design_name} not found in norms file")

        baseline_val = self.norms[design_name]["baseline"]
        baseline = torch.tensor([[baseline_val]], dtype=torch.float).to(self.device)

        # --- predict delta ---
        with torch.no_grad():
            delta = self.model(graph, recipe, lengths, baseline)

        pred_power = delta.item() + baseline_val
        return pred_power