Vocabulary= {'refactor -z': 1, 'balance': 2, 'rewrite': 3, 'rewrite -z': 4, 'resub': 5, 'resub -z': 6, 'refactor': 7}

def encode_recipe(recipe, vocab):
    """
    Encode a list of operations into a list of integers based on the provided vocabulary.
    
    Args:
        recipe (list of str): List of operations in the recipe
        vocab (dict): Dictionary mapping operation strings to integer indices
    
    Returns:
        list of int: Encoded recipe as a list of integers
    """
    return [vocab[op] for op in recipe if op in vocab]

if __name__ == "__main__":
    # Example usage
    recipe = ["rewrite", "balance", "rewrite", "refactor"]
    encoded = encode_recipe(recipe, Vocabulary)
    print("Encoded recipe:", encoded)