# Status: what is done, partial, and not supported

Verified on: Apple M5 (MPS backend) and CPU, macOS, Python 3.13, PyTorch 2.15.0,
transformers 5.23.0 (reference tests). **No CUDA GPU was available**; nothing below has been run on
CUDA. Treat CUDA-specific claims as untested.

Legend: ✅ implemented and tested · 🟡 partial / limited · ❌ not implemented

## Components

| Component | Status | Evidence |
|---|---|---|
| MHA / GQA / MQA (fused QKV, SDPA, native GQA) | ✅ | matches HF `LlamaForCausalLM` logits with mapped weights (3 layers, float64, ≤1e-6); SDPA vs eager reference; gradcheck |
| QK-Norm (RMSNorm or LayerNorm per head) | ✅ | unit tests |
| MLA (q-LoRA optional, decoupled RoPE, compressed cache, absorbed decode) | ✅ | matches HF `DeepseekV3Attention` (with/without q-LoRA), both full and absorbed-incremental paths |
| Mamba-2 (chunked SSD + recurrent decode, ngroups) | ✅ | matches HF `Mamba2Mixer` torch path (ngroups 1 and 2); chunked vs token recurrence; gradcheck |
| Gated DeltaNet (chunked WY + recurrent decode) | ✅ | kernel matches HF Qwen3-Next `torch_chunk/recurrent_gated_delta_rule`; matches the matrix-form definition; gradcheck. Layer wiring follows FLA but is **not** compared layer-for-layer against FLA |
| Kimi Delta Attention (per-channel decay; chunked with secondary tiles + doubling inverse, recurrent decode) | ✅ | kernel matches the token recurrence and the matrix-form definition; reduces to Gated DeltaNet with a shared decay; finite fwd/bwd at chunk 64 with every channel at `g_min`; gradcheck; MPS == CPU. Chunked algorithm written from scratch for MPS; **not** compared against FLA / Kimi Linear |
| Dense FFN (GELU / ReLU / SiLU / ReLU²), SwiGLU, GeGLU, generic GLU | ✅ | compared to explicit formulas |
| MoE: softmax/sigmoid routing, top-k, renormalisation, shared experts | ✅ | dropless dispatch == naive per-token loop |
| MoE: load-balancing loss, z-loss, aux-loss-free bias (DeepSeek-V3 style) | ✅ | value test at known routing; bias test shows load imbalance shrinking |
| MoE: capacity-factor dispatch with token dropping (static shapes) | ✅ | == dropless with ample capacity; exact drop fraction; compiles with 0 graph breaks |
| RMSNorm, LayerNorm, pre-norm and post-norm blocks | ✅ | |
| RoPE (incl. partial), ALiBi, learned absolute positions, NoPE | ✅ | relative-position property, ALiBi reference slopes, padded learned positions |
| Position/mixer compatibility errors | ✅ | 12 parametrized error tests |
| Standard and gated residuals | ✅ | |
| Hyper-connections (dynamic + static) | ✅ | n=1 == standard residual network; n=4 init == standard |
| mHC | 🟡 | formula + Sinkhorn constraints tested; init choices are ours; no comparison with DeepSeek's (unreleased) code; no fused kernels (the paper's 6.7% overhead figure relies on them) |
| Attention residuals (Full AttnRes) | 🟡 | from paper description; not yet compared with the official MoonshotAI code; Block AttnRes ❌ |
| Sliding-window / local attention | ❌ | |
| Packed sequences / varlen (document masking) | ❌ | left/right padding supported instead |

## Model / systems

| Feature | Status | Notes |
|---|---|---|
| Heterogeneous layers, `replace`, `replace_layers`, `every()` | ✅ | |
| JSON config round-trip, fingerprints, deterministic init | ✅ | init is seeded via a forked RNG and does not disturb the global RNG |
| Summary, parameter breakdown, active params (MoE), FLOP estimate | ✅ | FLOPs: matmul-only analytical model, see [flops.md](flops.md) |
| SVG / Mermaid architecture diagrams | ✅ | |
| Incremental decoding with per-mixer states | ✅ | any chunking == full pass, for every mixer and a 5-type hybrid |
| Batched generation with left padding | ✅ | == generating each prompt alone (greedy) |
| Sampling: temperature, top-k, top-p, EOS, seeded generator | ✅ | |
| Beam search, speculative decoding, paged KV cache | ❌ | |
| Training: AdamW, warmup + cosine/linear, grad accumulation, clipping | ✅ | |
| BF16 autocast | ✅ | CPU and MPS |
| FP16 + GradScaler | 🟡 | coded for CUDA only, untested (no CUDA) |
| Exact checkpoint resume (model, optimizer, scheduler, data sampler, RNG) | ✅ | resumed run equals uninterrupted run |
| Gradient checkpointing | ✅ | gradients equal to non-checkpointed |
| `torch.compile` | 🟡 | dense / attention models: 0 graph breaks, ~1.9× faster training on MPS. Dropless MoE has graph breaks (host sync for per-expert counts); use `capacity_factor` for compile. Recurrent mixers compile (CPU-tested) but were not benchmarked compiled |
| FlashAttention | 🟡 | via `F.scaled_dot_product_attention`, which selects FlashAttention on supported CUDA GPUs; the `flash-attn` package is not integrated; untested here |
| Triton / custom CUDA kernels | ❌ | Triton is unavailable on macOS. Mamba-2, Gated DeltaNet and KDA use pure-PyTorch chunked algorithms — correct, but slower than `mamba_ssm` / FLA kernels on GPU. Integrating those as optional backends is the next step |
| Grouped-GEMM MoE / expert parallelism | ❌ | dropless path loops over experts in Python |
| Distributed training (DDP / FSDP / TP) | ❌ | the model is a plain `nn.Module` and should wrap with DDP/FSDP, but this is untested |
| Production readiness | ❌ | not claimed |

## Measured results (this machine)
See the README for the training run, ablation table and overhead benchmark.
