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

# Step 10 - core_attention
def core_attention(q, cands, valid, sink):
    # q:     (B, n, n_heads, c)
    # cands: (B, n, L, c)
    # valid: (B, n, L)
    # sink:  (n_heads,)

    # Compute scaled dot-product attention logits.
    # Shape: (B, n, n_heads, L)
    logits = torch.einsum("bthc,btlc->bthl", q, cands)
    logits = logits / (q.shape[-1] ** 0.5)

    # Mask invalid candidates so that their attention probabilities are zero.
    logits = logits.masked_fill(
        ~valid.unsqueeze(2),
        float("-inf")
    )

    # Append one learnable attention-sink logit per head.
    # Shape: (B, n, n_heads, 1)
    sink_logits = sink.reshape(1, 1, -1, 1).expand(
        q.shape[0], q.shape[1], -1, -1
    )

    # Include the sink in softmax normalization.
    # The sink absorbs probability mass without contributing a value vector.
    all_logits = torch.cat((logits, sink_logits), dim=-1)
    all_probs = torch.softmax(all_logits, dim=-1)

    # Discard the sink probability; retain candidate probabilities.
    # Shape: (B, n, n_heads, L)
    probs = all_probs[..., :-1]

    # Compute the weighted sum of candidate values.
    # Shape: (B, n, n_heads, c)
    out = torch.einsum("bthl,btlc->bthc", probs, cands)

    return out, probs

# Step 11 - window_entries
def window_entries(x, n_win):
    # Validate the sliding-window size.
    if not isinstance(n_win, int) or isinstance(n_win, bool) or n_win <= 0:
        raise ValueError("n_win must be a positive integer")

    B, n, c = x.shape

    # Compute the token positions for each sliding window.
    # positions[t, j] = t - n_win + 1 + j
    # Shape: (n, n_win)
    positions = (
        torch.arange(n, device=x.device).unsqueeze(1)
        - n_win + 1
        + torch.arange(n_win, device=x.device).unsqueeze(0)
    )

    # An entry is valid if its position is within the sequence.
    # Shape: (n, n_win)
    valid = positions >= 0

    # Handle an empty input sequence.
    if n == 0:
        win = x.new_zeros((B, 0, n_win, c))
        return win, valid, positions

    # Replace negative positions with zero temporarily so that
    # indexing is safe. Invalid entries will be zeroed afterward.
    safe_positions = positions.clamp_min(0)

    # Gather the uncompressed KV entries for every sliding window.
    # Shape: (B, n, n_win, c)
    win = x[:, safe_positions, :]

    # Zero-pad entries that lie before the beginning of the sequence.
    win = win.masked_fill(
        ~valid.unsqueeze(0).unsqueeze(-1),
        0.0
    )

    return win, valid, positions

# Step 12 - apply_partial_rope
def apply_partial_rope(x, positions, rope_dim, base=10000.0):
    # Validate the rotary dimension.
    D = x.shape[-1]
    if not isinstance(rope_dim, int) or isinstance(rope_dim, bool):
        raise ValueError("rope_dim must be an integer")
    if rope_dim < 0 or rope_dim > D or rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even and between 0 and the last dimension")
    if base <= 0:
        raise ValueError("base must be positive")

    # Nothing to rotate if rope_dim is zero.
    if rope_dim == 0:
        return x

    half = rope_dim // 2

    # Preserve the dimensions outside the rotary portion.
    unchanged = x[..., :D - rope_dim]
    rotary = x[..., D - rope_dim:]

    # Use float32 for angle computation when inputs use reduced precision.
    calc_dtype = (
        torch.float32
        if x.dtype in (torch.float16, torch.bfloat16)
        else x.dtype
    )

    # Ensure positions are on the same device and broadcastable
    # to the leading dimensions of x.
    positions = torch.as_tensor(positions, device=x.device)
    positions = torch.broadcast_to(positions, x.shape[:-1])
    positions = positions.to(dtype=calc_dtype)

    # Compute inverse frequencies: base^(-2i / rope_dim).
    i = torch.arange(half, device=x.device, dtype=calc_dtype)
    inv_freq = base ** (-2.0 * i / rope_dim)

    # Each position receives one rotation angle per dimension pair.
    angles = positions.unsqueeze(-1) * inv_freq
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    # Pair dimensions using the half-split convention:
    # (0, half), (1, half+1), ..., (half-1, rope_dim-1).
    rotary_calc = rotary.to(dtype=calc_dtype)
    a = rotary_calc[..., :half]
    b = rotary_calc[..., half:]

    # Rotate each pair: (a, b) -> (a*cos - b*sin, a*sin + b*cos).
    rotated_a = a * cos_angles - b * sin_angles
    rotated_b = a * sin_angles + b * cos_angles

    # Restore the original input dtype and concatenate the untouched
    # dimensions with the rotated dimensions.
    rotated = torch.cat((rotated_a, rotated_b), dim=-1).to(dtype=x.dtype)

    return torch.cat((unchanged, rotated), dim=-1)

# Step 13 - GroupedOutputProjection
class GroupedOutputProjection(nn.Module):
    def __init__(self, n_heads, head_dim, groups, d_group, d_model):
        super().__init__()

        if groups <= 0 or n_heads % groups != 0:
            raise ValueError("groups must be positive and divide n_heads")

        # Store the number of groups and dimensions for each group.
        self.groups = groups
        self.group_in_dim = (n_heads * head_dim) // groups

        # Learnable projection for each group.
        # Shape: (groups, n_heads * head_dim // groups, d_group)
        self.group_weight = nn.Parameter(
            torch.randn(groups, self.group_in_dim, d_group)
            / math.sqrt(self.group_in_dim)
        )

        # Final bias-free projection back to the model dimension.
        self.final = nn.Linear(groups * d_group, d_model, bias=False)

    def forward(self, o):
        # o: (B, n, n_heads, head_dim)
        B, n, n_heads, head_dim = o.shape

        # Divide consecutive heads into groups and concatenate the
        # heads within each group into a single feature vector.
        # Shape: (B, n, groups, group_in_dim)
        grouped = o.reshape(B, n, self.groups, self.group_in_dim)

        # Apply the corresponding learnable projection to each group.
        # Shape: (B, n, groups, d_group)
        projected = torch.einsum(
            "bngd,gdo->bngo",
            grouped,
            self.group_weight
        )

        # Concatenate the projected groups.
        # Shape: (B, n, groups * d_group)
        concatenated = projected.reshape(B, n, self.groups * self.group_weight.shape[-1])

        # Project to the model dimension.
        # Shape: (B, n, d_model)
        return self.final(concatenated)

# Step 14 - rope_core_attention
def rope_core_attention(q, q_pos, cands, valid, cand_pos, sink, rope_dim):
    # q:        (B, n, n_heads, c)
    # q_pos:    (n,)
    # cands:    (B, n, L, c)
    # valid:    (B, n, L)
    # cand_pos: broadcastable to (B, n, L)
    # sink:     (n_heads,)

    B, n, n_heads, c = q.shape

    # Reshape query positions so they broadcast across the batch
    # and attention heads.
    q_positions = torch.as_tensor(q_pos, device=q.device)
    q_positions = q_positions.reshape(1, n, 1)

    # Apply partial RoPE to the queries using their absolute positions.
    q_rotated = apply_partial_rope(
        q, q_positions, rope_dim
    )

    # Apply partial RoPE to candidate KV entries using their positions.
    # cand_pos may be (B, n, L) or another broadcastable shape.
    cands_rotated = apply_partial_rope(
        cands, cand_pos, rope_dim
    )

    # Run shared KV multi-query attention with attention sinks.
    # The sink contributes to normalization but not to the output values.
    out, _ = core_attention(
        q_rotated,
        cands_rotated,
        valid,
        sink
    )

    # Undo the query's absolute-position rotation to obtain the
    # position-relative output representation.
    out = apply_partial_rope(
        out, -q_positions, rope_dim
    )

    # Shape: (B, n, n_heads, c)
    return out

# Step 15 - CSALayer
def csa_core(layer, Hq, q_pos, comp, kidx, visible, win, win_valid, win_pos):
    # Hq: (B, nq, d_model)
    # comp: (B, nb, head_dim)
    # kidx: (B, nb, idx_dim)
    # win: (B, nq, n_win, head_dim)

    B, nq, _ = Hq.shape

    # Generate attention queries and indexer queries from the shared latent.
    q, q_idx = layer.queries(Hq)

    # Normalize attention queries per head.
    q = layer.q_norm(q)

    # Compute weighted multi-head ReLU index scores.
    # idx_weight(Hq) produces per-token weights for indexer heads.
    w_idx = layer.idx_weight(Hq)
    scores = index_scores(q_idx, w_idx, kidx)

    # Ensure the visibility mask is on the correct device.
    visible = visible.to(device=scores.device, dtype=torch.bool)

    # Select the top-k visible compressed entries for every query.
    idx, idx_valid = select_topk(scores, visible, layer.top_k)
    k_eff = idx.shape[-1]

    # Gather the selected compressed KV entries.
    # Expand comp across queries, then gather along the block axis.
    head_dim = comp.shape[-1]
    nb = comp.shape[1]

    expanded_comp = comp.unsqueeze(1).expand(B, nq, nb, head_dim)
    gather_idx = idx.unsqueeze(-1).expand(B, nq, k_eff, head_dim)
    selected_comp = torch.gather(expanded_comp, dim=2, index=gather_idx)

    # A compressed entry represents the final token of its block.
    # Invalid selections are ignored by idx_valid.
    selected_pos = layer.m * (idx + 1) - 1

    # Move window tensors to the query device.
    win = win.to(device=q.device)
    win_valid = win_valid.to(device=q.device, dtype=torch.bool)
    win_pos = win_pos.to(device=q.device, dtype=torch.long)
    idx_valid = idx_valid.to(device=q.device)

    # Expand window validity and positions across the batch dimension.
    win_valid_b = win_valid.unsqueeze(0).expand(B, -1, -1)
    win_pos_b = win_pos.unsqueeze(0).expand(B, -1, -1)

    # Combine selected compressed entries with uncompressed local entries.
    # Candidate ordering: selected compressed entries, then sliding window.
    cands = torch.cat((selected_comp, win), dim=2)

    # Combine validity masks and absolute candidate positions.
    valid = torch.cat((idx_valid, win_valid_b), dim=2)
    cand_pos = torch.cat((selected_pos, win_pos_b), dim=2)

    # Apply partial RoPE, run sink-aware attention, and counter-rotate
    # the resulting head outputs using the query positions.
    out = rope_core_attention(
        q,
        q_pos,
        cands,
        valid,
        cand_pos,
        layer.sink,
        layer.rope_dim
    )

    # Project grouped head outputs back to the model dimension.
    return layer.out_proj(out)


class CSALayer(nn.Module):
    def __init__(
        self, d_model, head_dim, n_heads, d_c, idx_dim,
        n_idx_heads, m, n_win, top_k, rope_dim, groups, d_group
    ):
        super().__init__()

        # Store the configuration required by CSA and the shared helper.
        self.m = m
        self.n_win = n_win
        self.top_k = top_k
        self.rope_dim = rope_dim
        self.head_dim = head_dim

        # Compress hidden states into KV entries and indexer keys.
        self.kv_comp = TokenCompressor(d_model, head_dim, m)
        self.idx_comp = TokenCompressor(d_model, idx_dim, m)

        # Generate attention and indexer queries from one shared latent.
        self.queries = LatentQueries(
            d_model, d_c, n_heads, head_dim, n_idx_heads, idx_dim
        )

        # Generate per-token weights for the indexer heads.
        self.idx_weight = nn.Linear(
            d_model, n_idx_heads, bias=False
        )

        # Project hidden states into uncompressed local KV entries.
        self.win_kv = nn.Linear(
            d_model, head_dim, bias=False
        )

        # Per-head RMSNorm for queries and KV entries.
        self.q_norm = HeadRMSNorm(head_dim)
        self.kv_norm = HeadRMSNorm(head_dim)

        # Learnable attention-sink logit for every attention head.
        self.sink = nn.Parameter(torch.zeros(n_heads))

        # Grouped projection of the concatenated attention-head outputs.
        self.out_proj = GroupedOutputProjection(
            n_heads, head_dim, groups, d_group, d_model
        )

    def forward(self, H):
        # H: (B, n, d_model)
        B, n, _ = H.shape

        # Build compressed KV entries and normalize them per head.
        comp = self.kv_norm(self.kv_comp(H))

        # Build compressed indexer keys.
        kidx = self.idx_comp(H)

        # Build the local sliding window from normalized KV entries.
        local_kv = self.kv_norm(self.win_kv(H))
        win, win_valid, win_pos = window_entries(
            local_kv, self.n_win
        )

        # Allow queries to attend only to fully preceding compressed blocks.
        nb = comp.shape[1]
        visible = causal_block_mask(n, nb, self.m).to(device=H.device)

        # Absolute position of each query token.
        q_pos = torch.arange(n, device=H.device)

        # Run compressed sparse attention and return model-width outputs.
        return csa_core(
            self,
            H,
            q_pos,
            comp,
            kidx,
            visible,
            win,
            win_valid,
            win_pos
        )

# Step 16 - hca_compress
def hca_compress(C, Z, bias, m):
    # Split the KV entries and compression logits into non-overlapping
    # blocks of m tokens. Incomplete trailing tokens are discarded.
    C_blocks, _ = split_blocks(C, m)
    Z_blocks, _ = split_blocks(Z, m)

    # Add the learnable positional bias to every block.
    # Shapes: (B, nb, m, c) + (m, c)
    logits = Z_blocks + bias

    # Normalize compression weights over the m positions
    # independently for each channel and block.
    weights = torch.softmax(logits, dim=2)

    # Compute the weighted sum of KV entries within each block.
    # Output shape: (B, n // m, c)
    compressed = (weights * C_blocks).sum(dim=2)

    return compressed

