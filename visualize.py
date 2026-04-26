import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
ABC_STATS_DIR = 'abcStats'
ANNEAL_RESULTS_DIR = 'anneal_results'
EXCLUDED_ALIASES = {'src_rw', 'src_rs', 'src_rws', 'resyn2rs', 'compress2rs', 'c2rs', '&resyn2rs', '&compress2rs','choice2'}
INCLUDE_ONLY_ALIASES = {'resyn','resyn2','resyn3','resyn2a','compress','compress2'}  # Set to a set of aliases to include, or None to include all except excluded
def main():
    # Find all CSV files in the abcStats folder
    csv_files = glob.glob(os.path.join(ABC_STATS_DIR, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {ABC_STATS_DIR}")
        return

    designs = []
    abc_powers_list = []
    anneal_powers_list = []
    diff_texts = []
    improvements = []

    # Process each design
    for csv_path in csv_files:
        # Extract base name (e.g., 'designA' from 'abcStats/designA.csv')
        filename = os.path.basename(csv_path)
        design_name = os.path.splitext(filename)[0]
        
        # Expected JSON path (e.g., 'anneal_results/designA_result.json')
        json_path = os.path.join(ANNEAL_RESULTS_DIR, f"{design_name}_results.json")
        
        if not os.path.exists(json_path):
            print(f"Warning: Skipping {design_name} - no matching JSON found at {json_path}")
            continue

        # 1. Read ABC Stats
        try:
            df = pd.read_csv(csv_path)
            if 'power' not in df.columns:
                print(f"Warning: 'power' column missing in {csv_path}")
                continue
            # Filter out rows with excluded aliases
            if 'alias' in df.columns:
                df = df[~df['alias'].isin(EXCLUDED_ALIASES)]
                # df = df[df['alias'].isin(INCLUDE_ONLY_ALIASES)]

            # Convert to numeric, dropping any empty or malformed rows
            powers = pd.to_numeric(df['power'], errors='coerce').dropna().tolist()
            if not powers:
                continue
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        # 2. Read Anneal Result
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                anneal_power = float(data.get('real_power', np.nan))
                if np.isnan(anneal_power):
                    continue
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            continue

        # 3. Calculate Median and Differences
        median_abc = np.median(powers)
        
        # Calculate how much better/worse it is
        # Assuming lower power is better: 
        # Negative % means Anneal is lower (Better). Positive % means Anneal is higher (Worse).
        pct_diff = ((anneal_power - median_abc) / median_abc) * 100
        if pct_diff < 0:
            status = f"{abs(pct_diff):.1f}% better"
            color = 'green' # Just tracking color for the text annotation if you want to use it
        else:
            status = f"{abs(pct_diff):.1f}% worse"
            color = 'red'

        # Store data for sorting
        improvements.append({
            'design': design_name,
            'powers': powers,
            'anneal_power': anneal_power,
            'status': status,
            'color': color,
            'pct_diff': pct_diff
        })

    if not improvements:
        print("No valid data pairs found to plot.")
        return

    # Sort by pct_diff (ascending - most negative/best improvements first) and keep top 6
    improvements.sort(key=lambda x: x['pct_diff'])
    improvements = improvements[:6]

    # Extract sorted data for plotting
    for imp in improvements:
        designs.append(imp['design'])
        abc_powers_list.append(imp['powers'])
        anneal_powers_list.append(imp['anneal_power'])
        diff_texts.append((imp['status'], imp['color']))

    # --- Plotting ---
    plt.figure(figsize=(14, 7))

    for i, design in enumerate(designs):
        # Plot all ABC Stats as blue scatter points
        x_vals = [i] * len(abc_powers_list[i])
        plt.scatter(x_vals, abc_powers_list[i], color='royalblue', alpha=0.4, 
                    label='ABC Stats (recipes)' if i == 0 else "")
        
        # Plot the Anneal Result as a solid red dot
        plt.scatter([i], [anneal_powers_list[i]], color='red', zorder=5, s=80, 
                    edgecolors='black', label='Anneal Result' if i == 0 else "")
        
        # Annotate the Better/Worse percentage above the data points
        # Find the highest point in this column to place the text nicely above it
        max_y_in_column = max(max(abc_powers_list[i]), anneal_powers_list[i])
        offset = max_y_in_column * 0.02 # 2% padding above the highest point
        
        text_str, text_color = diff_texts[i]
        plt.text(i, max_y_in_column + offset, text_str, ha='center', va='bottom', 
                 fontsize=9, color=text_color, fontweight='bold', rotation=45)

    # Formatting the Graph
    plt.xticks(range(len(designs)), designs, rotation=45, ha='right')
    plt.ylabel('Power', fontweight='bold')
    plt.xlabel('Designs', fontweight='bold')
    plt.title('ABC Stats vs. Anneal Results (Power Analysis)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to make room for x-axis labels and top annotations
    plt.tight_layout() 
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()