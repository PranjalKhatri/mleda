import pandas as pd

def load_design_csv(power_csv, stats_csv):
    power_df = pd.read_csv(power_csv)   # sid, area, power
    stats_df = pd.read_csv(stats_csv)   # sid, BUFF, NOT, AND, PI, PO, LP

    df = pd.merge(power_df, stats_df, on="sid")
    return df