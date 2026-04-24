import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


# -------------------------
# Graph Encoder
# -------------------------
class AIGEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINEConv(nn1)
        self.conv2 = GINEConv(nn1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        return global_mean_pool(x, batch)


# -------------------------
# Recipe Encoder (FIXED)
# -------------------------
class RecipeEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size + 1,   # +1 for padding
            emb_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, seq, lengths):
        emb = self.embedding(seq)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (h, _) = self.lstm(packed)
        return h[-1]


# -------------------------
# Stats Encoder
# -------------------------
class StatsEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, stats):
        return self.mlp(stats)


# -------------------------
# Full Model
# -------------------------
class PowerPredictor(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.graph_encoder = AIGEncoder(in_dim=5, hidden_dim=128)
        self.recipe_encoder = RecipeEncoder(vocab_size, emb_dim=32, hidden_dim=64)
        self.stats_encoder = StatsEncoder(in_dim=6)

        self.head = nn.Sequential(
            nn.Linear(128 + 64 + 32 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data, recipe, lengths, stats, baseline):
        g = self.graph_encoder(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

        r = self.recipe_encoder(recipe, lengths)
        s = self.stats_encoder(stats)

        x = torch.cat([g, r, s, baseline], dim=1)
        return self.head(x)