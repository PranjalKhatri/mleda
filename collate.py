import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    graphs = [item['graph'] for item in batch]
    recipes = [item['recipe'] for item in batch]

    stats = torch.stack([item['stats'] for item in batch])
    baseline = torch.stack([item['baseline'] for item in batch])
    target = torch.stack([item['target'] for item in batch])

    # --- recipe lengths (important for LSTM) ---
    lengths = torch.tensor([len(r) for r in recipes], dtype=torch.long)

    # --- padding (use -1 to avoid clash with vocab) ---
    recipes_padded = pad_sequence(
        recipes,
        batch_first=True,
        padding_value=-1
    )

    # --- replace padding with 0 for embedding lookup ---
    # (we'll use mask to ignore it)
    recipes_padded_clipped = recipes_padded.clone()
    recipes_padded_clipped[recipes_padded_clipped == -1] = 0

    # --- PyG batching ---
    from torch_geometric.data import Batch
    graph_batch = Batch.from_data_list(graphs)

    return (
        graph_batch,
        recipes_padded_clipped,
        lengths,          # NEW
        stats,
        baseline,
        target
    )