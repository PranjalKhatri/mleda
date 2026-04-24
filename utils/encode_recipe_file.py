#!/usr/bin/env python3
"""
Script to read a recipe file and encode it.
Each line in the recipe file contains one operation.
"""

import sys
from pathlib import Path
from recipe_encoder import encode_recipe, Vocabulary


def encode_recipe_from_file(recipe_file):
    """
    Read a recipe file and encode it.
    
    Args:
        recipe_file (str): Path to the recipe file
    
    Returns:
        list: Encoded recipe as integers
    """
    if not Path(recipe_file).exists():
        print(f"Error: File '{recipe_file}' not found")
        sys.exit(1)
    
    # Read recipe from file - each line is one operation
    with open(recipe_file, 'r') as f:
        recipe = [line.strip() for line in f if line.strip()]
    
    print(f"\n=== Recipe Encoding for {Path(recipe_file).name} ===")
    print(f"Original recipe operations:")
    for i, op in enumerate(recipe, 1):
        print(f"  {i}. {op}")
    
    # Encode the recipe
    encoded = encode_recipe(recipe, Vocabulary)
    
    print(f"\nEncoded recipe:")
    print(f"  {encoded}")
    
    print(f"\nEncoding details:")
    for op, code in zip(recipe, encoded):
        print(f"  '{op}' -> {code}")
    
    return encoded


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python encode_recipe_file.py <recipe_file>")
        print("Example: python encode_recipe_file.py data/scripts/script0.txt")
        sys.exit(1)
    
    recipe_file = sys.argv[1]
    encode_recipe_from_file(recipe_file)
