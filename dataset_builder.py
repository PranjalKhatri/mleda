import torch
from torch.utils.data import Dataset
import copy


def normalize_df(df):
    cols = ['BUFF', 'NOT', 'AND', 'PI', 'PO', 'LP']
    df[cols] = (df[cols] - df[cols].mean()) / (df[cols].std() + 1e-6)
    return df


class PowerDataset(Dataset):
    def __init__(self, designs, recipe_dict):
        """
        designs: list of dicts {
            'graph': pyg_data,
            'df': merged dataframe
        }
        """
        self.samples = []

        for design in designs:
            graph = design['graph']
            df = design['df'].copy()

            # normalize stats
            df = normalize_df(df)

            # --- baseline handling ---
            if 0 not in df['sid'].values:
                raise ValueError("sid=0 not found for baseline. Fix your dataset.")

            baseline_power = df[df['sid'] == 0]['power'].values[0]

            for _, row in df.iterrows():
                sid = int(row['sid'])

                if sid not in recipe_dict:
                    continue  # skip if missing recipe

                # --- delta target ---
                delta_power = row['power'] - baseline_power

                sample = {
                    # IMPORTANT: deepcopy graph to avoid PyG batch corruption
                    "graph": copy.deepcopy(graph),

                    "recipe": torch.tensor(
                        recipe_dict[sid],
                        dtype=torch.long
                    ),

                    "stats": torch.tensor(
                        [
                            row['BUFF'],
                            row['NOT'],
                            row['AND'],
                            row['PI'],
                            row['PO'],
                            row['LP']
                        ],
                        dtype=torch.float
                    ),

                    "baseline": torch.tensor(
                        [baseline_power],
                        dtype=torch.float
                    ),

                    # target is now DELTA
                    "target": torch.tensor(
                        [delta_power],
                        dtype=torch.float
                    )
                }

                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]