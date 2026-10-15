# DeepSeek-V4 Hybrid Attention from Scratch

Build the attention stack that lets DeepSeek-V4 run million-token contexts, in PyTorch, from the technical report's equations. Start with the token-level compressor that folds every m tokens into one KV entry with overlapping softmax weights and learnable positional biases, and the per-head RMSNorm the report applies to queries and KV entries. Add the lightning indexer and the top-k selector of DeepSeek Sparse Attention, then the shared key-value multi-query attention where each compressed entry is both key and value, with attention sinks. Attach the sliding-window branch, partial RoPE with the position-minus-i counter-rotation of the outputs, and the grouped output projection to finish Compressed Sparse Attention (CSA). Build Heavily Compressed Attention (HCA) with its larger rate m' and dense attention over compressed entries, interleave the two into a small decoder stack, then build the heterogeneous KV cache that decodes one token at a time with mixed BF16/FP8 entry storage, verify it reproduces the full forward pass, and quantify the KV-cache and FLOPs savings against a GQA8 baseline at one million tokens.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** split_blocks
- [x] **2.** overlap_windows
- [x] **3.** compress_blocks
- [x] **4.** TokenCompressor
- [x] **5.** HeadRMSNorm
- [x] **6.** LatentQueries
- [x] **7.** index_scores
- [x] **8.** causal_block_mask
- [x] **9.** select_topk
- [x] **10.** core_attention
- [x] **11.** window_entries
- [x] **12.** apply_partial_rope
- [x] **13.** GroupedOutputProjection

---

Built on Deep-ML.
