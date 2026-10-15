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

# Step 6 - LatentQueries
class LatentQueries(nn.Module):
    def __init__(self, d_model, d_c, n_heads, head_dim, n_idx_heads, idx_dim):
        super().__init__()

        # Store the attention and indexer dimensions.
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_idx_heads = n_idx_heads
        self.idx_dim = idx_dim

        # Project hidden states into a shared compressed latent.
        self.down = nn.Linear(d_model, d_c, bias=False)

        # Generate attention queries from the shared latent.
        self.up = nn.Linear(d_c, n_heads * head_dim, bias=False)

        # Generate indexer queries from the same shared latent.
        self.up_idx = nn.Linear(d_c, n_idx_heads * idx_dim, bias=False)

    def forward(self, H):
        # H: (B, n, d_model)
        # Compress hidden states into the shared latent.
        latent = self.down(H)  # (B, n, d_c)

        # Project and reshape attention queries.
        q = self.up(latent)
        B, n, _ = q.shape
        q = q.reshape(B, n, self.n_heads, self.head_dim)

        # Project and reshape indexer queries using the same latent.
        q_idx = self.up_idx(latent)
        q_idx = q_idx.reshape(B, n, self.n_idx_heads, self.idx_dim)

        return q, q_idx

# Step 7 - index_scores
def index_scores(q_idx, w_idx, k_idx):
    # q_idx: (B, n, n_idx_heads, idx_dim)
    # w_idx: (B, n, n_idx_heads)
    # k_idx: (B, nb, idx_dim)

    # Compute dot products between every query head and every compressed key.
    # Result shape: (B, n, n_idx_heads, nb)
    scores = torch.einsum("bthd,bsd->bths", q_idx, k_idx)

    # Apply ReLU to the dot products.
    scores = torch.relu(scores)

    # Weight each indexer head's scores and sum across heads.
    # Result shape: (B, n, nb)
    scores = (scores * w_idx.unsqueeze(-1)).sum(dim=2)

    return scores

# Step 8 - causal_block_mask
def causal_block_mask(n, nb, m):
    # Validate the block size.
    if not isinstance(m, int) or isinstance(m, bool) or m <= 0:
        raise ValueError("m must be a positive integer")

    # Determine which compressed block contains each query token.
    # Shape: (n,)
    query_blocks = torch.arange(n) // m

    # Indices of the compressed KV blocks.
    # Shape: (nb,)
    block_indices = torch.arange(nb)

    # A query token may attend only to blocks strictly earlier
    # than the block containing that token.
    # Broadcasting produces a boolean mask of shape (n, nb).
    mask = block_indices.unsqueeze(0) < query_blocks.unsqueeze(1)

    return mask

# Step 9 - select_topk
def select_topk(scores, visible, k):
    # Validate k.
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a non-negative integer")

    B, n, nb = scores.shape
    k_eff = min(k, nb)

    # Expand the visibility mask across the batch dimension.
    # visible: (n, nb) -> (B, n, nb)
    visible_batch = visible.unsqueeze(0).expand(B, -1, -1)

    # Exclude invisible blocks by assigning their scores negative infinity.
    masked_scores = scores.masked_fill(
        ~visible_batch, float("-inf")
    )

    # Stable descending sort preserves the original block-index order
    # for tied scores, so lower block indices win ties.
    sorted_scores, sorted_idx = torch.sort(
        masked_scores,
        dim=-1,
        descending=True,
        stable=True
    )

    # Keep at most k_eff block indices per query.
    idx = sorted_idx[..., :k_eff]

    # A selected entry is valid exactly when its block was visible.
    valid = torch.gather(visible_batch, dim=-1, index=idx)

    # Replace invalid indices with zero as required.
    idx = torch.where(valid, idx, torch.zeros_like(idx))

    return idx, valid

