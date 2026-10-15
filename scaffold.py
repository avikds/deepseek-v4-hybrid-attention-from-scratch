"""
DeepSeek-V4 Hybrid Attention from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""DeepSeek-V4 Hybrid Attention from Scratch (DeepSeek-AI technical report, sections 2.3 and 3.5.1).

Story: compress every m tokens into one KV entry with the overlapping softmax compressor; let the
lightning indexer pick top-k entries per query and attend with shared key-value MQA, sinks, a sliding
window and partial RoPE with the position-minus-i counter-rotation (CSA); compress m' tokens at a time
and attend densely (HCA); interleave both into a tiny decoder stack; decode one token at a time from a
heterogeneous KV cache with mixed BF16/FP8 storage and verify it matches the full forward pass; and put
numbers on the cache and FLOPs savings against a BF16 GQA8 baseline at one million tokens.
"""
import time
import torch


def main() -> None:
    torch.manual_seed(0)
    t0 = time.time()
    d_model, vocab = 64, 50
    csa = dict(head_dim=32, n_heads=4, d_c=32, idx_dim=16, n_idx_heads=2, m=4, n_win=8, top_k=4, rope_dim=16, groups=2, d_group=32)
    hca = dict(head_dim=32, n_heads=4, d_c=32, m=16, n_win=8, rope_dim=16, groups=2, d_group=32)
    model = HybridStack(vocab, d_model, n_layers=3, hca_every=3, csa_kwargs=csa, hca_kwargs=hca)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"hybrid stack: d_model {d_model}, layers {model.kinds}, m {csa['m']}, m' {hca['m']}, window {csa['n_win']}, top-k {csa['top_k']}, {n_params} parameters")

    # ---- 1. Token-level compression ----
    n = 40
    tok = torch.randint(0, vocab, (1, n))
    H = model.blocks[0].norm1(model.embed(tok))
    layer0 = model.blocks[0].attn
    comp = layer0.kv_comp(H)
    H_cut = H.clone(); H_cut[:, 8:] = 0.0
    same = torch.allclose(layer0.kv_comp(H_cut)[:, :2], comp[:, :2], atol=1e-6)
    kc = layer0.kv_comp
    S = torch.softmax(overlap_windows(kc.z_a(H), kc.z_b(H), 4, float('-inf')) + torch.cat([kc.bias_a, kc.bias_b]), dim=2)
    print(f"\n1. Compressor: {n} tokens -> {comp.shape[1]} CSA entries (m=4) and {model.blocks[2].attn.kv_norm(hca_compress(model.blocks[2].attn.kv_proj(H), model.blocks[2].attn.z_proj(H), model.blocks[2].attn.bias, 16)).shape[1]} HCA entries (m'=16), tail of {n - 16 * (n // 16)} tokens for HCA")
    print(f"   entries 0-1 depend only on tokens 0-7: {same}; window weights sum to one per channel: {torch.allclose(S.sum(2), torch.ones_like(S.sum(2)))}; block 0 gives zero weight to its padded half: {float(S[0, 0, 4:].sum()):.1f}")

    # ---- 2. Indexer and sparse selection ----
    q, q_idx = layer0.queries(H)
    scores = index_scores(q_idx, layer0.idx_weight(H), layer0.idx_comp(H))
    visible = causal_block_mask(n, comp.shape[1], 4)
    idx, valid = select_topk(scores, visible, 4)
    print(f"\n2. Lightning indexer: score matrix {tuple(scores.shape)}; token 39 may see {int(visible[39].sum())} blocks and keeps {int(valid[0, 39].sum())}: blocks {sorted(idx[0, 39].tolist())}")
    print(f"   tokens 0-3 see no compressed block ({int(valid[0, :4].sum())} valid slots); the window branch covers them")

    # ---- 3. CSA properties ----
    out = layer0(H)
    H2 = H.clone(); H2[:, 20:] = torch.randn(1, n - 20, d_model)
    causal = torch.allclose(out[:, :20], layer0(H2)[:, :20], atol=1e-6)
    qn = layer0.q_norm(q)
    comp_n = layer0.kv_norm(comp)
    win, wv, wp = window_entries(layer0.kv_norm(layer0.win_kv(H)), 8)
    sel = torch.gather(comp_n[:, None].expand(1, n, comp.shape[1], 32), 2, idx[..., None].expand(1, n, 4, 32))
    cands = torch.cat([sel, win], 2); cvalid = torch.cat([valid, wv[None]], 2)
    _, probs = core_attention(qn, cands, cvalid, layer0.sink)
    with torch.no_grad():
        layer0.sink.fill_(3.0)
    _, probs_sink = core_attention(qn, cands, cvalid, layer0.sink)
    with torch.no_grad():
        layer0.sink.zero_()
    print(f"\n3. CSA layer: output {tuple(out.shape)}, causal under a change of tokens 20-39: {causal}")
    print(f"   mean total attention per head with sink logit 0: {float(probs.sum(-1).mean()):.3f}; with sink logit 3: {float(probs_sink.sum(-1).mean()):.3f} (sinks let heads attend to nothing)")
    pos_a = rope_core_attention(qn[:, :8], torch.arange(8), cands[:, :8], cvalid[:, :8], torch.cat([4 * (idx[:, :8] + 1) - 1, wp[None, :8]], 2), layer0.sink, 16)
    pos_b = rope_core_attention(qn[:, :8], torch.arange(8) + 1000, cands[:, :8], cvalid[:, :8], torch.cat([4 * (idx[:, :8] + 1) - 1, wp[None, :8]], 2) + 1000, layer0.sink, 16)
    print(f"   partial RoPE + counter-rotation: shifting every position by 1000 changes the outputs by {float((pos_a - pos_b).abs().max()):.1e} (relative positions only)")

    # ---- 4. Full stack and a short training run ----
    period = torch.randint(0, vocab, (5,))
    data = period.repeat(16)[None]  # a periodic sequence of 80 tokens
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for step in range(60):
        logits = model(data[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), data[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    print(f"\n4. Hybrid stack trained 60 steps on a period-5 token sequence: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    # ---- 5. Incremental decoding from the heterogeneous KV cache ----
    model.eval()
    tok = torch.randint(0, vocab, (2, n))
    with torch.no_grad():
        full = model(tok)
        inc, caches = prefill(model, tok)
        incq, caches_q = prefill(model, tok, quantize=True)
    print(f"\n5. KV cache: decoding {n} tokens one at a time vs the full forward pass: max |logit diff| {float((full - inc).abs().max()):.1e} (exact storage), {float((full - incq).abs().max()):.1e} (BF16/FP8 storage)")
    for i, c in enumerate(caches):
        nb = 0 if c.comp is None else c.comp.shape[1]
        tail = 0 if c.tail is None else c.tail.shape[1]
        extra = f", {c.kidx.shape[1]} indexer keys" if c.kind == 'csa' else ""
        print(f"   layer {i} ({c.kind}, m={c.layer.m}): {nb} compressed entries{extra}, {tail} tail tokens buffered, window of {c.window.shape[1]} at positions {c.win_pos[0]}-{c.win_pos[-1]}")
    entry = store_kv_entry(caches[0].comp, 16)
    print(f"   CSA entry storage: {entry_bytes(entry) // caches[0].comp.shape[1] // 2} bytes per entry mixed BF16/FP8 vs {caches[0].comp.shape[2] * 2} pure BF16")
    with torch.no_grad():
        nxt = decode_step(model, caches, tok[:, -1], n)
        ref = model(torch.cat([tok, tok[:, -1:]], 1))[:, n]
    print(f"   one more decode step: max |diff| vs a longer forward pass {float((nxt - ref).abs().max()):.1e}; layer 0 now holds {caches[0].comp.shape[1]} entries")

    # ---- 6. Cost report at one million tokens ----
    cfg = dict(head_dim=512, idx_dim=128, n_heads=64, n_idx_heads=64, m=4, m_prime=16, n_win=128, top_k=1024, rope_dim=64, csa_layers=30, hca_layers=30)
    print("\n6. Cost model, assumed production-scale config (the report gives no numbers): c=512, c_I=128, 64 heads, m=4, m'=16, window 128, top-k 1024, 30 CSA + 30 HCA layers")
    for ctx in (2 ** 17, 2 ** 20):
        rep = attention_cost_report(cfg, ctx)
        print(f"   {ctx // 1024}K tokens: KV cache {rep['hybrid_bytes'] / 2 ** 30:.2f} GB vs GQA8 BF16 {rep['baseline_bytes'] / 2 ** 30:.1f} GB = {100 * rep['bytes_ratio']:.1f}%; attention FLOPs per token {rep['hybrid_flops'] / 1e9:.1f} vs {rep['baseline_flops'] / 1e9:.1f} GFLOP = {100 * rep['flops_ratio']:.1f}%")
    print("   report: about 2% of the GQA8 cache at 1M tokens; its 27% FLOPs figure is whole-model vs DeepSeek-V3.2, not attention vs GQA8")
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

