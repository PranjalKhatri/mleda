#!/usr/bin/env python3
"""
Analyze all abcStats CSV files and rank recipes per design with normalized metrics.
Lower power values are better.
"""

import os
import csv
from pathlib import Path
from collections import defaultdict
import statistics

def analyze_recipes_per_design(stats_dir='abcStats'):
    """
    Analyze each design file separately and compute per-design rankings.
    
    Args:
        stats_dir: Directory containing the CSV files
        
    Returns:
        Dictionary with design names as keys and recipe rankings as values
    """
    design_data = {}
    
    # Get all CSV files
    csv_files = sorted(Path(stats_dir).glob('*.csv'))
    
    print(f"Found {len(csv_files)} design files\n")
    
    # Parse each CSV file
    for csv_file in csv_files:
        design_name = csv_file.stem
        recipes = {}
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    alias = row['alias'].strip()
                    power = float(row['power'])
                    recipes[alias] = power
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
            continue
        
        if recipes:
            # Calculate average power for this design
            avg_power = statistics.mean(recipes.values())
            
            # Calculate percentage improvement for each recipe
            recipe_improvements = {}
            for alias, power in recipes.items():
                # Percentage improvement = (avg - power) / avg * 100
                # Positive means better than average
                pct_improvement = ((avg_power - power) / avg_power) * 100
                recipe_improvements[alias] = {
                    'power': power,
                    'avg_design_power': avg_power,
                    'pct_improvement': pct_improvement
                }
            
            design_data[design_name] = {
                'avg_power': avg_power,
                'recipes': recipe_improvements
            }
    
    return design_data

def print_top_5_per_design(design_data):
    """
    Print top 5 recipes for each design ranked by percentage improvement.
    """
    print("\n" + "="*100)
    print("TOP 5 RECIPES PER DESIGN (Ranked by % Improvement from Design Average)")
    print("="*100)
    
    for design_name in sorted(design_data.keys()):
        data = design_data[design_name]
        avg_power = data['avg_power']
        recipes = data['recipes']
        
        # Sort by percentage improvement (descending - higher is better)
        sorted_recipes = sorted(recipes.items(), key=lambda x: x[1]['pct_improvement'], reverse=True)
        
        print(f"\n{design_name.upper()}")
        print(f"  Design Average Power: {avg_power:.2f}")
        print(f"  {'-'*95}")
        print(f"  {'Rank':<6} {'Recipe':<20} {'Power':<15} {'% Improvement':<20}")
        print(f"  {'-'*95}")
        
        for rank, (alias, metrics) in enumerate(sorted_recipes[:5], 1):
            print(f"  {rank:<6} {alias:<20} {metrics['power']:<15.2f} {metrics['pct_improvement']:>18.2f}%")

def compute_overall_rankings(design_data):
    """
    Compute overall recipe rankings using normalized metrics across all designs.
    Metric: average percentage improvement across all designs.
    """
    recipe_metrics = defaultdict(list)
    
    # Collect improvements for each recipe across all designs
    for design_name, data in design_data.items():
        recipes = data['recipes']
        for alias, metrics in recipes.items():
            recipe_metrics[alias].append(metrics['pct_improvement'])
    
    # Calculate aggregate metrics for each recipe
    overall_stats = {}
    for alias, improvements in recipe_metrics.items():
        overall_stats[alias] = {
            'avg_improvement': statistics.mean(improvements),
            'min_improvement': min(improvements),
            'max_improvement': max(improvements),
            'consistency': statistics.stdev(improvements) if len(improvements) > 1 else 0,
            'num_designs': len(improvements)
        }
    
    return overall_stats

def print_overall_rankings(overall_stats):
    """
    Print overall recipe rankings by average percentage improvement.
    """
    sorted_recipes = sorted(overall_stats.items(), key=lambda x: x[1]['avg_improvement'], reverse=True)
    
    print("\n" + "="*100)
    print("OVERALL RECIPE RANKINGS (Normalized Across All Designs)")
    print("="*100)
    print(f"{'Rank':<6} {'Recipe':<20} {'Avg % Improvement':<20} {'Min %':<15} {'Max %':<15} {'Consistency':<15}")
    print("-"*100)
    
    for rank, (alias, stats) in enumerate(sorted_recipes, 1):
        print(f"{rank:<6} {alias:<20} {stats['avg_improvement']:>18.2f}% "
              f"{stats['min_improvement']:>13.2f}% {stats['max_improvement']:>13.2f}% {stats['consistency']:>13.2f}")
    
    print("\n" + "="*100)
    print("TOP 10 RECIPES (Best Overall Performance)")
    print("="*100)
    for rank, (alias, stats) in enumerate(sorted_recipes[:10], 1):
        print(f"{rank:2d}. {alias:<20} Avg Improvement: {stats['avg_improvement']:>7.2f}%")
    
    return sorted_recipes

if __name__ == '__main__':
    # Analyze recipes per design
    design_data = analyze_recipes_per_design('abcStats')
    
    if not design_data:
        print("No design data found!")
        exit(1)
    
    # Print top 5 per design
    print_top_5_per_design(design_data)
    
    # Compute and print overall rankings
    overall_stats = compute_overall_rankings(design_data)
    sorted_recipes = print_overall_rankings(overall_stats)
    
    # Save results to file
    output_file = 'recipe_rankings_normalized.txt'
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("OVERALL RECIPE RANKINGS (Normalized Across All Designs)\n")
        f.write("="*100 + "\n")
        f.write(f"{'Rank':<6} {'Recipe':<20} {'Avg % Improvement':<20} {'Min %':<15} {'Max %':<15} {'Consistency':<15}\n")
        f.write("-"*100 + "\n")
        
        for rank, (alias, stats) in enumerate(sorted_recipes, 1):
            f.write(f"{rank:<6} {alias:<20} {stats['avg_improvement']:>18.2f}% "
                   f"{stats['min_improvement']:>13.2f}% {stats['max_improvement']:>13.2f}% {stats['consistency']:>13.2f}\n")
    
    print(f"\nResults saved to {output_file}")
