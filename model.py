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

# Step 2 - overlap_windows
def overlap_windows(xa, xb, m, pad_value):
    # Validate the block size.
    if not isinstance(m, int) or isinstance(m, bool) or m <= 0:
        raise ValueError("m must be a positive integer")

    B, n, c = xa.shape
    nb = n // m
    full_length = nb * m

    # Split both sequences into complete blocks; ignore trailing tokens.
    a_blocks = xa[:, :full_length, :].reshape(B, nb, m, c)
    b_blocks = xb[:, :full_length, :].reshape(B, nb, m, c)

    # Handle sequences shorter than one complete block.
    if nb == 0:
        return xa.new_empty((B, 0, 2 * m, c))

    # Prepend a padding block to the b-series, then shift its blocks
    # so window i receives block i-1 from xb.
    pad_block = xb.new_full((B, 1, m, c), pad_value)
    previous_b_blocks = torch.cat(
        (pad_block, b_blocks[:, :-1, :, :]),
        dim=1
    )

    # Concatenate each a-block with the corresponding previous b-block.
    windows = torch.cat(
        (a_blocks, previous_b_blocks),
        dim=2
    )

    return windows

