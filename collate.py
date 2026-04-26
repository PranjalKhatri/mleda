import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch


def collate_fn(batch):
    graphs  = [item['graph']  for item in batch]
    recipes = [item['recipe'] for item in batch]
    targets = [item['target'] for item in batch]

    lengths = torch.tensor([len(r) for r in recipes], dtype=torch.long)

    recipes_padded = pad_sequence(
        recipes, batch_first=True, padding_value=0
    )

    graph_batch = Batch.from_data_list(graphs)
    target      = torch.stack(targets)          # (B, 1)

    return graph_batch, recipes_padded, lengths, target