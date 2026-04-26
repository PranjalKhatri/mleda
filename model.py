import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool


# -------------------------
# Graph Encoder
# -------------------------
class AIGEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=128):
        super().__init__()

        self.edge_encoders = nn.ModuleList([
            nn.Linear(1, in_dim),
            nn.Linear(1, hidden_dim),
            nn.Linear(1, hidden_dim),
        ])

        dims = [in_dim, hidden_dim, hidden_dim]
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(
                nn.Linear(dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            for i in range(3)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])

        # mean + add pooling concatenated → 2*hidden_dim
        self.out_dim = hidden_dim * 2

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, norm, edge_enc in zip(self.convs, self.norms, self.edge_encoders):
            e = edge_enc(edge_attr)
            x = F.relu(norm(conv(x, edge_index, e)))

        mean_pool = global_mean_pool(x, batch)          # (B, H)
        add_pool  = global_add_pool(x, batch)           # (B, H)
        return torch.cat([mean_pool, add_pool], dim=1)  # (B, 2H)


# -------------------------
# Recipe Encoder
# Bidirectional 2-layer LSTM + attention
# -------------------------
class RecipeEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size + 1, emb_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.15,
            bidirectional=True      # captures left-to-right and right-to-left op deps
        )
        self.attn   = nn.Linear(hidden_dim * 2, 1)
        self.out_dim = hidden_dim * 2   # 256

    def forward(self, seq, lengths):
        emb = self.embedding(seq)                           # (B, T, emb)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)                          # (B, T, 2H)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        scores  = self.attn(out).squeeze(-1)                # (B, T)
        scores  = scores.masked_fill(seq == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)             # (B, T)
        context = (out * weights.unsqueeze(-1)).sum(dim=1)  # (B, 2H)
        return context


# -------------------------
# Full Model — NO baseline input
# Inputs: AIG graph + recipe
# Output: raw predicted power
# -------------------------
class PowerPredictor(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.graph_encoder  = AIGEncoder(in_dim=5, hidden_dim=128)
        self.recipe_encoder = RecipeEncoder(vocab_size, emb_dim=64, hidden_dim=128)

        g_dim = self.graph_encoder.out_dim    # 256
        r_dim = self.recipe_encoder.out_dim   # 256

        # gate: recipe features scaled by graph context
        # forces the two encoders to interact rather than one dominating
        self.gate = nn.Sequential(
            nn.Linear(g_dim + r_dim, r_dim),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(g_dim + r_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, data, recipe, lengths):
        g = self.graph_encoder(
            data.x, data.edge_index, data.edge_attr, data.batch
        )                                            # (B, 256)

        r = self.recipe_encoder(recipe, lengths)     # (B, 256)

        gate = self.gate(torch.cat([g, r], dim=1))   # (B, 256)
        r    = r * gate                              # gated recipe

        x = torch.cat([g, r], dim=1)                # (B, 512)
        return self.head(x)                          # (B, 1)