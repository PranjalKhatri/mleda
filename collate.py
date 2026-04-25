import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch


def collate_fn(batch):
    graphs = [item['graph'] for item in batch]
    recipes = [item['recipe'] for item in batch]

    baseline = torch.stack([item['baseline'] for item in batch])
    target = torch.stack([item['target'] for item in batch])

    # --- recipe lengths ---
    lengths = torch.tensor([len(r) for r in recipes], dtype=torch.long)

    # --- padding (0 is safe: matches embedding padding_idx=0) ---
    recipes_padded = pad_sequence(
        recipes,
        batch_first=True,
        padding_value=0
    )

    # --- PyG batching ---
    graph_batch = Batch.from_data_list(graphs)

    return (
        graph_batch,
        recipes_padded,
        lengths,
        baseline,
        target
    )