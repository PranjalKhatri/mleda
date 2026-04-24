"""
Script to analyze CSV files and extract top 25% of power values.
"""

import pandas as pd
import sys
from pathlib import Path


def analyze_power(csv_file, power_column='power'):
    """
    Read a CSV file and return top 25% of power values.
    
    Args:
        csv_file (str): Path to the CSV file
        power_column (str): Name of the column containing power values
    
    Returns:
        pd.DataFrame: DataFrame containing top 25% of records by power value
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if power column exists
    if power_column not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Column '{power_column}' not found in CSV")
    
    # Calculate 75th percentile (top 25%)
    percentile_75 = df[power_column].quantile(0.75)
    
    # Filter records with power >= 75th percentile
    top_25_percent = df[df[power_column] >= percentile_75].sort_values(
        by=power_column, ascending=False
    )
    
    print(f"\n=== Power Analysis for {Path(csv_file).name} ===")
    print(f"Total records: {len(df)}")
    print(f"Top 25% threshold (75th percentile): {percentile_75:.4f}")
    print(f"Records in top 25%: {len(top_25_percent)}")
    print(f"\nTop 25% Power Values Statistics:")
    print(f"  Min: {top_25_percent[power_column].min():.4f}")
    print(f"  Max: {top_25_percent[power_column].max():.4f}")
    print(f"  Mean: {top_25_percent[power_column].mean():.4f}")
    print(f"  Median: {top_25_percent[power_column].median():.4f}")
    
    print(f"\nTop 25% Records:")
    # print(top_25_percent.to_string())
    
    return top_25_percent


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python power_analyzer.py <csv_file> [power_column]")
        print("Example: python power_analyzer.py data/power/aes_power.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    power_column = sys.argv[2] if len(sys.argv) > 2 else 'power'
    
    if not Path(csv_file).exists():
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    
    analyze_power(csv_file, power_column)
