import torch
from torch.utils.data import Dataset
import copy


class PowerDataset(Dataset):
    def __init__(self, designs, recipe_dict):
        """
        designs: list of dicts {
            'graph': pyg_data,
            'df':    merged dataframe with 'sid' and 'Power' columns,
            'name':  design_name
        }
        recipe_dict: sid -> list of token ids
        """
        self.samples = []

        for design in designs:
            graph = design['graph']
            df    = design['df'].copy()

            for _, row in df.iterrows():
                sid = int(row['sid'])

                if sid not in recipe_dict:
                    continue

                self.samples.append({
                    "graph": copy.deepcopy(graph),

                    "recipe": torch.tensor(
                        recipe_dict[sid],
                        dtype=torch.long
                    ),

                    # raw power — no baseline subtraction, no normalisation
                    # train.py handles per-batch normalisation
                    "target": torch.tensor(
                        [row['Power']],
                        dtype=torch.float
                    ),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]