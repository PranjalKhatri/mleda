import torch
from torch.utils.data import Dataset
import copy


# -------------------------
# Normalize per design
# -------------------------
def normalize_df(df):
    cols = ['PI', 'PO', 'AND', 'edges', 'Level']

    mean = df[cols].mean()
    std = df[cols].std() + 1e-6

    df[cols] = (df[cols] - mean) / std

    return df, mean, std


class PowerDataset(Dataset):
    def __init__(self, designs, recipe_dict):
        """
        designs: list of dicts {
            'graph': pyg_data,
            'df': merged dataframe,
            'name': design_name
        }
        """
        self.samples = []

        for design in designs:
            graph = design['graph']
            df = design['df'].copy()

            # -------------------------
            # Normalize stats (design-wise)
            # -------------------------
            df, mean, std = normalize_df(df)

            # -------------------------
            # Baseline (top 25%)
            # -------------------------
            baseline_power = df['Power'].quantile(0.75)

            # -------------------------
            # Build samples
            # -------------------------
            for _, row in df.iterrows():
                sid = int(row['sid'])

                if sid not in recipe_dict:
                    continue

                delta_power = row['Power'] - baseline_power

                sample = {
                    # deepcopy required for PyG safety
                    "graph": copy.deepcopy(graph),

                    "recipe": torch.tensor(
                        recipe_dict[sid],
                        dtype=torch.long
                    ),

                    # ✅ ONLY ABC STATS
                    "stats": torch.tensor(
                        [
                            row['PI'],
                            row['PO'],
                            row['AND'],    # (acts like ND proxy)
                            row['edges'],
                            row['Level']
                        ],
                        dtype=torch.float
                    ),

                    "baseline": torch.tensor(
                        [baseline_power],
                        dtype=torch.float
                    ),

                    # delta target
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