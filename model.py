"""
DeepSeek-V4 Hybrid Attention from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - split_blocks
import torch

def split_blocks(x, m):
    # Validate the block size.
    if not isinstance(m, int) or isinstance(m, bool) or m <= 0:
        raise ValueError("m must be a positive integer")

    B, n, c = x.shape
    nb = n // m
    full_length = nb * m

    # Split the complete blocks from the trailing remainder.
    blocks = x[:, :full_length, :].reshape(B, nb, m, c)
    tail = x[:, full_length:, :]

    return blocks, tail

