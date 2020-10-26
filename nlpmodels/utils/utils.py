import numpy as np
import torch
import torch.nn as nn
from typing import List


def set_seed_everywhere():
    """
    Function to setting seeds everywhere.
    """
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_cosine_similar(target_word : str, word_to_idx : dict, embeddings : torch.Tensor) -> List:
    """
    Function for calculating cosine similarities across dictionary versus target word, descending order.
        Args:
            target_word (str): token of interest
            word_to_idx (dict): word to index map
            embeddings (torch.Tensor): embedding vectors
        Returns:
            returns a list of (token, cosine_similarity) pairs.
    """
    cos = nn.CosineSimilarity(dim=0)
    word_embedding = embeddings[word_to_idx[target_word.lower()]]
    distances = []
    for word, index in word_to_idx.items():
        if word == "<MASK>" or word == target_word or word=="<UNK>":
            continue
        distances.append((word, cos(word_embedding, embeddings[index])))

    results = sorted(distances, key=lambda x: x[1],reverse=True)
    return results