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
- [x] **14.** rope_core_attention
- [x] **15.** CSALayer
- [x] **16.** hca_compress
- [x] **17.** HCALayer
- [x] **18.** layer_kinds
- [x] **19.** HybridBlock
- [x] **20.** HybridStack
- [x] **21.** store_kv_entry
- [x] **22.** LayerCache
- [x] **23.** cached_attention
- [x] **24.** decode_step
- [x] **25.** attention_cost_report

## Results

```
hybrid stack: d_model 64, layers ['csa', 'csa', 'hca'], m 4, m' 16, window 8, top-k 4, 138188 parameters

1. Compressor: 40 tokens -> 10 CSA entries (m=4) and 2 HCA entries (m'=16), tail of 8 tokens for HCA
   entries 0-1 depend only on tokens 0-7: True; window weights sum to one per channel: True; block 0 gives zero weight to its padded half: 0.0

2. Lightning indexer: score matrix (1, 40, 10); token 39 may see 9 blocks and keeps 4: blocks [0, 3, 4, 8]
   tokens 0-3 see no compressed block (0 valid slots); the window branch covers them

3. CSA layer: output (1, 40, 64), causal under a change of tokens 20-39: True
   mean total attention per head with sink logit 0: 0.912; with sink logit 3: 0.411 (sinks let heads attend to nothing)
   partial RoPE + counter-rotation: shifting every position by 1000 changes the outputs by 1.6e-05 (relative positions only)

4. Hybrid stack trained 60 steps on a period-5 token sequence: loss 4.467 -> 0.007

5. KV cache: decoding 40 tokens one at a time vs the full forward pass: max |logit diff| 3.6e-06 (exact storage), 1.1e-01 (BF16/FP8 storage)
   layer 0 (csa, m=4): 9 compressed entries, 9 indexer keys, 4 tail tokens buffered, window of 8 at positions 32-39
   layer 1 (csa, m=4): 9 compressed entries, 9 indexer keys, 4 tail tokens buffered, window of 8 at positions 32-39
   layer 2 (hca, m=16): 2 compressed entries, 8 tail tokens buffered, window of 8 at positions 32-39
   CSA entry storage: 48 bytes per entry mixed BF16/FP8 vs 64 pure BF16
   one more decode step: max |diff| vs a longer forward pass 9.5e-07; layer 0 now holds 10 entries

6. Cost model, assumed production-scale config (the report gives no numbers): c=512, c_I=128, 64 heads, m=4, m'=16, window 128, top-k 1024, 30 CSA + 30 HCA layers
   128K tokens: KV cache 0.78 GB vs GQA8 BF16 30.0 GB = 2.6%; attention FLOPs per token 53.4 vs 257.7 GFLOP = 20.7%
   1024K tokens: KV cache 6.22 GB vs GQA8 BF16 240.0 GB = 2.6%; attention FLOPs per token 391.6 vs 2061.6 GFLOP = 19.0%
   report: about 2% of the GQA8 cache at 1M tokens; its 27% FLOPs figure is whole-model vs DeepSeek-V3.2, not attention vs GQA8

done in 2.4s
```
