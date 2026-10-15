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

# Step 3 - compress_blocks
def compress_blocks(Ca, Cb, Za, Zb, Ba, Bb, m):
    # Validate the block size.
    if not isinstance(m, int) or isinstance(m, bool) or m <= 0:
        raise ValueError("m must be a positive integer")

    B, n, c = Ca.shape

    # Repeat positional biases for each block, aligning them with
    # the within-block token positions.
    bias_repeats = (n + m - 1) // m
    Ba_expanded = Ba.repeat(bias_repeats, 1)[:n]
    Bb_expanded = Bb.repeat(bias_repeats, 1)[:n]

    # Add learnable positional biases to the compression logits.
    Za_biased = Za + Ba_expanded.unsqueeze(0)
    Zb_biased = Zb + Bb_expanded.unsqueeze(0)

    # Build overlapping windows of logits.
    # Missing entries in the first b-window receive -inf so that
    # their softmax weights are exactly zero.
    logits = overlap_windows(
        Za_biased, Zb_biased, m, pad_value=float("-inf")
    )

    # Build matching windows of KV entries.
    # Missing entries are zero-padded.
    values = overlap_windows(
        Ca, Cb, m, pad_value=0.0
    )

    # Normalize across the 2m window positions independently
    # for each channel.
    weights = torch.softmax(logits, dim=2)

    # Compute the weighted sum to produce one compressed KV entry
    # per complete block.
    compressed = (weights * values).sum(dim=2)

    return compressed

# Step 4 - TokenCompressor
import torch.nn as nn

class TokenCompressor(nn.Module):
    def __init__(self, d_model, c, m):
        super().__init__()

        # Store the compression block size.
        self.m = m

        # Project hidden states into the two KV-entry series
        # and their corresponding compression-logit series.
        self.kv_a = nn.Linear(d_model, c, bias=False)
        self.kv_b = nn.Linear(d_model, c, bias=False)
        self.z_a = nn.Linear(d_model, c, bias=False)
        self.z_b = nn.Linear(d_model, c, bias=False)

        # Learnable positional biases for the two token series.
        # Initialize both biases to zero.
        self.bias_a = nn.Parameter(torch.zeros(m, c))
        self.bias_b = nn.Parameter(torch.zeros(m, c))

    def forward(self, H):
        # Project hidden states into KV entries and compression logits.
        Ca = self.kv_a(H)
        Cb = self.kv_b(H)
        Za = self.z_a(H)
        Zb = self.z_b(H)

        # Compress each complete block using overlapping softmax
        # weights and learnable positional biases.
        compressed = compress_blocks(
            Ca, Cb, Za, Zb,
            self.bias_a, self.bias_b,
            self.m
        )

        # Shape: (B, n // m, c). Incomplete trailing tokens
        # do not produce a compressed entry.
        return compressed

# Step 5 - HeadRMSNorm
class HeadRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        # Store the numerical stability constant.
        self.eps = eps

        # Learnable per-dimension gain, initialized to ones.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute the mean squared value along the last dimension.
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # Normalize using the reciprocal square root of the RMS variance.
        x_norm = x * torch.rsqrt(mean_square + self.eps)

        # Apply the learnable gain, preserving the input shape.
        return x_norm * self.weight

