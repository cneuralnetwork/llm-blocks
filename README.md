<div align="center">

# 🧱 llm-blocks

**LEGO-style building blocks for LLM architecture research in PyTorch.**

Swap attention for MLA, Mamba-2 or Gated DeltaNet, a dense FFN for MoE, or a plain residual for
hyper-connections, by changing **one line**. The rest of the model, the training loop and
generation stay the same.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5%2B-ee4c2c)
![Tests: 248 passing](https://img.shields.io/badge/tests-248%20passing-brightgreen)
![Status: research preview](https://img.shields.io/badge/status-research%20preview-orange)

</div>

```python
from llm_blocks import nn

model = nn.LLM(
    vocab_size=32000,
    dim=768,
    layers=[
        nn.Block(
            mixer=nn.GQA(heads=12, kv_heads=4, position=nn.RoPE()),   # ← swap me: nn.MLA, nn.Mamba2, nn.GatedDeltaNet
            ffn=nn.SwiGLU(hidden_dim=2048),                           # ← swap me: nn.MoE(...)
            norm=nn.RMSNorm(),
        )
        for _ in range(12)
    ],
)
print(model.summary())   # per-layer table · parameter breakdown · active params · FLOPs/token
```

<p align="center">
  <img src="docs/assets/hybrid_architecture.svg" width="640"
       alt="Block diagram of a hybrid model: Mamba-2 layers with SwiGLU, and every fourth layer GQA with a 16-expert MoE">
  <br><sub>A hybrid Mamba-2 + GQA + MoE model, drawn by <code>model.visualize()</code></sub>
</p>

---

**Contents:**
[Why](#why) ·
[Install](#install) ·
[Tour](#a-two-minute-tour) ·
[Components](#components) ·
[Ablations](#run-an-ablation) ·
[Training and generation](#training-and-generation) ·
[Custom blocks](#add-your-own-block) ·
[How it works](#how-it-works) ·
[Correctness](#correctness) ·
[Performance](#performance) ·
[Demo](#run-the-demo) ·
[Status](#status-and-limitations)

## Why

Modern LLMs are assembled from interchangeable parts, but most codebases hard-wire them together.
Changing one part usually means rewriting the model, the KV cache, the generation loop and the
config. That slows research down and makes ablations hard to trust.

llm-blocks makes these questions one-line experiments that you can actually run:

| Question | Code |
|---|---|
| What happens if I replace GQA with MLA? | `base.replace(mixer=nn.MLA(heads=12))` |
| …if every fourth attention layer is Mamba-2? | `base.replace_layers(nn.every(4, start=3), mixer=nn.Mamba2())` |
| …if the dense FFN becomes MoE? | `base.replace(ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=1024), num_experts=16, top_k=2))` |
| …if I change the residual mechanism? | `base.replace(residual=nn.MHC(n=4))` |

Every variant trains with the same `Trainer`, generates with the same `generate()`, saves to the
same JSON format and reports comparable parameter, FLOP and throughput numbers.

## Install

```bash
git clone https://github.com/cneuralnetwork/llm-blocks && cd llm-blocks
pip install -e .                # core: only needs torch>=2.5
pip install -e ".[dev]"         # + pytest
pip install -e ".[reference]"   # + transformers, for the reference-implementation tests
```

## A two-minute tour

**1. Describe a base model and derive variants.** Configs are frozen, comparable and serializable.

```python
base = nn.LLMConfig(vocab_size=32000, dim=768, depth=12,
                    mixer=nn.GQA(heads=12, kv_heads=4), ffn=nn.SwiGLU(hidden_dim=2048))

hybrid = base.replace_layers(nn.every(4, start=3), mixer=nn.Mamba2())
moe    = base.replace(ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=1024), num_experts=16, top_k=2, shared_experts=1))

hybrid.save("hybrid.json"); assert nn.LLMConfig.load("hybrid.json") == hybrid
hybrid.fingerprint()   # stable hash for experiment tracking
```

**2. Build and inspect.**

```python
model = nn.LLM(config=hybrid)
print(model.architecture())
```
```
LLM(vocab_size=32000, dim=768, params=131.24M)
├─ embed: Embedding(32000, 768)
├─ layers[0-2] × 3: Block(pre-norm RMSNorm)
│    ├─ mixer:    GQA(heads=12, kv_heads=4)
│    ├─ ffn:      SwiGLU(hidden_dim=2048)
│    └─ residual: Residual()
├─ layers[3]: Block(pre-norm RMSNorm)
│    ├─ mixer:    Mamba2()
│    ├─ ffn:      SwiGLU(hidden_dim=2048)
│    └─ residual: Residual()
│  …
└─ lm_head: Linear(768 → 32000)
```

```python
nn.LLM(config=moe).num_parameters()          # 549,489,408 total
nn.LLM(config=moe).num_active_parameters()   # 153,127,680 used per token
model.flops_per_token(2048)                  # analytical forward FLOPs (see docs/flops.md)
model.visualize("arch.svg")                  # SVG block diagram, or fmt="mermaid"
```

**3. Generate.** It works for any mix of mixers, because each layer keeps its own decoding state:

```python
tokens = model.generate(input_ids, max_new_tokens=128, temperature=0.8, top_p=0.9)
```

Mistakes fail when the model is built, with an explanation, instead of silently doing the wrong
thing:

```python
nn.LLM(vocab_size=32000, dim=768, layers=[nn.Block(mixer=nn.Mamba2(position=nn.RoPE()))])
# SpecError: Mamba2 does not support RoPE (kind 'rotary'). Supported position kinds for Mamba2: ['none'].
```

## Components

| Family | Blocks |
|---|---|
| **Sequence mixers** | `MHA` · `MQA` · `GQA` · `MLA` (DeepSeek-V2 latent attention, compressed cache, absorbed decoding) · `Mamba2` · `GatedDeltaNet` · `KDA` ([Kimi Delta Attention](https://arxiv.org/abs/2510.26692), per-channel decay; chunked algorithm written from scratch to run on MPS) |
| **Feed-forward** | `FFN` · `GeluFFN` · `ReluFFN` · `GLU` · `SwiGLU` · `GeGLU` · `MoE`: top-k, softmax or sigmoid routing, shared experts, dropless or capacity dispatch, load-balance and z-losses, aux-loss-free bias balancing |
| **Normalization** | `RMSNorm` · `LayerNorm` · `QKNorm` (`qk_norm=True`) · pre- or post-norm blocks |
| **Positions** | `RoPE` (incl. partial) · `ALiBi` · `LearnedPositions` · `NoPE`, each checked for compatibility with its mixer |
| **Residuals** | `Residual` · `GatedResidual` · `HyperConnections` · `MHC` (manifold-constrained HC) · `AttnRes` (attention residuals) |
| **Training** | AdamW · warmup + cosine · gradient accumulation · clipping · BF16 · `torch.compile` · activation checkpointing · exact resume |
| **Research tools** | `summary()` · `parameter_breakdown()` · `num_active_parameters()` · `flops_per_token()` · `run_ablation()` · benchmarks · SVG and Mermaid diagrams |

The math for each residual mechanism is in [docs/residuals.md](docs/residuals.md).

## Run an ablation

`run_ablation` trains every variant from scratch with the same data order, optimizer, schedule,
validation windows and init seed:

```python
from llm_blocks.training import TrainConfig, run_ablation, results_table

results = run_ablation(
    {"gqa": base, "mla": base.replace(mixer=nn.MLA(heads=12)), "hybrid": hybrid, "moe": moe},
    train_data, val_data,
    TrainConfig(max_steps=2000, batch_size=32, seq_len=512, precision="bf16"),
    out_dir="runs/ablation",      # per-variant JSON: config, fingerprint, loss curves
)
print(results_table(results))
```

Here is what [`examples/ablation.py`](examples/ablation.py) produces on a laptop: 8 variants of a
2.5M-parameter byte-level model, one component changed per variant.

<p align="center">
  <img src="docs/assets/ablation_val_loss.svg" width="680" alt="Bar chart of final validation loss for 8 ablation variants">
</p>

<details>
<summary><b>Full table and how to read it</b></summary>

| variant | params | active | GFLOPs/tok (fwd) | train loss | val loss | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| baseline_gqa | 2.46M | 2.46M | 0.0054 | 1.861 | 1.735 | 44,133 |
| mla | 2.89M | 2.89M | 0.0064 | 1.934 | 1.998 | 40,692 |
| hybrid_mamba2_every2 | 2.88M | 2.88M | 0.0062 | 1.468 | 1.377 | 29,207 |
| all_gated_deltanet | 3.07M | 3.07M | 0.0064 | 1.396 | 1.329 | 25,340 |
| moe_8x_top2 | 7.78M | 2.47M | 0.0054 | 1.847 | 1.742 | 19,003 |
| gated_residual | 2.46M | 2.46M | 0.0054 | 1.853 | 1.737 | 41,366 |
| mhc_n4 | 2.69M | 2.69M | 0.0054 | 1.860 | 1.725 | 14,770 |
| attn_res | 2.46M | 2.46M | 0.0054 | 1.821 | 1.697 | 14,816 |

Setup: 6 layers, dim 192, Python standard-library source as the corpus, 300 steps × 16 × 256
tokens (1.2M tokens), bf16 on an Apple M5 GPU.

**This is a single-seed smoke test, not evidence about which architecture is better.** The variants
are not parameter-matched. At this tiny, byte-level scale the short convolutions inside Mamba-2 and
Gated DeltaNet are a strong local prior, which likely explains their faster early drop. MLA's
dimensions were derived automatically, not tuned. The tok/s column reflects plain-PyTorch kernels
on a laptop GPU and says nothing about optimized GPU kernels.
</details>

## Training and generation

```python
from llm_blocks.training import ByteTokenizer, TokenDataset, TrainConfig, Trainer

data = TokenDataset(ByteTokenizer().encode(text), seq_len=256)
train, val = data.split(0.05)

trainer = Trainer(nn.LLM(config=hybrid), train, val,
                  TrainConfig(max_steps=1000, batch_size=32, seq_len=256, lr=2e-3,
                              precision="bf16", grad_accum_steps=2, checkpoint_dir="runs/hybrid"))
trainer.fit()

model = Trainer.load_model("runs/hybrid/last.pt")    # rebuilds the architecture from the saved config
model.generate(prompt_ids, max_new_tokens=200, temperature=0.8, top_p=0.9)
```

- Batched prompts of different lengths are left-padded and passed with `attention_mask`.
- `do_sample=False` gives greedy decoding.
- `use_cache=False` recomputes the full sequence every step (useful as a reference).
- `model.save_pretrained(dir)` and `nn.LLM.from_pretrained(dir)` store the config and weights.

## Add your own block

A new mixer is two small classes: a module that does the computation and owns its decoding state,
and a spec that builds it. Subclassing a spec registers it, so configs using it serialize like
built-in ones. No library code changes:

```python
@dataclass(frozen=True, kw_only=True)
class CausalConv(nn.MixerSpec):
    kernel_size: int = 64
    def build(self, ctx):                       # ctx.dim, ctx.layer_idx, ctx.max_seq_len, ...
        return CausalConvMixer(ctx.dim, self.kernel_size)

cfg = base.replace_layers([1, 2], mixer=CausalConv(kernel_size=16))
```

Full working example, including incremental decoding: [examples/custom_mixer.py](examples/custom_mixer.py).

## How it works

- **Specs vs. modules.** `nn.GQA(...)` is a description, not a layer. `nn.LLM` builds each spec
  into an ordinary `torch.nn.Module`, supplying the model width and layer index. One spec can be
  reused across layers, and every build gets fresh weights.
- **One interface, different internals.** Every mixer implements
  `forward(x, state=None, padding_mask=None)` and `init_state(...)`. Their states are deliberately
  different:

  | Mixer | Decoding state | Size per sequence |
  |---|---|---|
  | MHA / GQA / MQA | pre-allocated K/V cache | `2 · kv_heads · head_dim · T` |
  | MLA | normalized latent + shared rotary key | `(kv_lora_rank + rope_dim) · T` |
  | Mamba-2 | conv window + SSM state | constant |
  | Gated DeltaNet | conv window + fast-weight matrix | constant |
  | KDA | conv window + fast-weight matrix | constant |

- **Auxiliary outputs without hidden state.** MoE balance losses travel through a per-forward
  `ForwardContext` and come back in `LMOutput.aux_loss`.
- **Residuals that change the stream.** Hyper-connections widen the residual stream to `n` copies,
  and attention residuals keep all previous layer outputs. The model checks that every block agrees
  on the stream type.

Details: [docs/design.md](docs/design.md).

## Correctness

`pytest` runs **248 tests in about 12 s** on CPU and Apple MPS.

- **Matches trusted reference implementations** (HuggingFace transformers, float64, agreement
  ≤1e-6):
  - a GQA/RoPE/SwiGLU model reproduces `LlamaForCausalLM` logits;
  - `MLA` reproduces `DeepseekV3Attention`, in both the full and the absorbed decoding paths;
  - `Mamba2` reproduces `Mamba2Mixer`;
  - the gated delta rule reproduces Qwen3-Next's chunked and recurrent kernels.
- **Incremental decoding equals the full causal pass** for every mixer, under any chunking, and for
  a 5-type hybrid model. Cached and uncached generation give identical tokens, and left-padded
  batches match one-prompt-at-a-time generation.
- **Also covered:** float64 gradient checks, causality, padding, BF16 vs FP32, CPU vs MPS,
  gradient-checkpointing equivalence, exact checkpoint resume, gradient-accumulation equivalence,
  serialization round-trips and error messages.

## Performance

The abstraction should cost nothing, so it is measured against a hand-written, monolithic PyTorch
Llama ([benchmarks/reference_llama.py](benchmarks/reference_llama.py)) with **identical weights**.
The fp32 logits differ by exactly 0. Setup: 54.5M parameters, batch 8 × 512 tokens, Apple M5 GPU,
median times:

| | llm-blocks | hand-written | ratio |
|---|---:|---:|---:|
| train step, bf16 | 467 ms | 475 ms | 0.98 |
| train step, bf16 + `torch.compile` | 245 ms | 246 ms | 0.99 |
| train step, fp32 | 735 ms | 736 ms | 1.00 |
| prefill, 128 tokens | 5.8 ms | 5.8 ms | 1.01 |
| decode, bf16 | 3.83 ms/token | 3.75 ms/token | 1.02 |

Training shows no measurable overhead. Decoding is 0–6% slower per token at batch size 1, from
Python bookkeeping of the state objects. Reproduce with `python benchmarks/compare_reference.py`;
raw numbers are in `benchmarks/results/`.

Where the speed comes from:
- a fused QKV projection;
- `scaled_dot_product_attention` with no materialized causal mask (FlashAttention on supported
  CUDA GPUs) and native GQA;
- fused gate/up projections in GLU layers;
- `F.rms_norm`;
- pre-allocated caches;
- static-shape MoE dispatch when `capacity_factor` is set (compiles with zero graph breaks).

## Run the demo

```bash
python examples/quickstart.py                 # build, inspect, derive variants, draw architecture.svg
```
```bash
python examples/train_tiny.py --steps 600     # train ~4M params on Python source, reload, generate
```
```bash
python examples/train_tiny.py --mixer mamba2  # same script, different architecture
```
```bash
python examples/custom_mixer.py               # plug in a user-defined mixer
```
```bash
python examples/ablation.py --steps 300       # 8-variant ablation (about 8 min on an M-series Mac)
```

`train_tiny.py` takes about 2.5 minutes on an Apple M5. Validation loss falls from 2.00 to 1.26
nats/byte. The reloaded checkpoint reproduces the logits exactly, and cached and uncached greedy
decoding agree.

## Status and limitations

This is a **research preview**. Everything listed above is implemented and tested, but it has only
run on Apple Silicon and CPU. Known gaps:

- **Not run on CUDA yet.** FlashAttention via SDPA, fused AdamW and fp16 loss scaling are untested.
- **No Triton or fused kernels for Mamba-2 and Gated DeltaNet.** The pure-PyTorch chunked
  algorithms are correct but slower than `mamba_ssm` and flash-linear-attention on GPU.
- **No distributed training** (DDP, FSDP or tensor parallelism).
- **Dropless MoE** loops over experts and breaks `torch.compile` graphs. Use `capacity_factor` to
  avoid this.
- **mHC and AttnRes** follow their papers, but are not yet compared against official code.
- **Not included:** sliding-window attention, packed sequences, beam search, speculative decoding.

Exact per-feature status and evidence: **[docs/STATUS.md](docs/STATUS.md)**.

<details>
<summary><b>Repository layout</b></summary>

```
llm_blocks/
  configs/spec.py         Spec base class, BuildContext, JSON registry
  nn/base.py              SequenceMixer, FeedForward, MixerState, ForwardContext
  nn/attention/           MHA / GQA / MQA (+ KV cache), MLA (+ latent cache)
  nn/mixers/              Mamba-2, Gated DeltaNet, KDA, causal conv
  nn/ffn/  nn/moe/        dense and gated FFNs, stacked experts, MoE
  nn/norm/ nn/position/   norms; RoPE, ALiBi, learned, NoPE
  nn/residual/            Residual, GatedResidual, HyperConnections, MHC, AttnRes
  nn/blocks/ nn/models/   Block, LLMConfig, LLM
  kernels/                pure-PyTorch SSD and gated-delta-rule kernels
  training/               data, Trainer, ablation runner
  inference/              generation and sampling
  utils/                  summaries, FLOPs, benchmarks
  visualization/          SVG and Mermaid diagrams
examples/  benchmarks/  tests/  docs/
```
</details>

## Acknowledgements

The Mamba-2 SSD algorithm follows [state-spaces/mamba](https://github.com/state-spaces/mamba)
(Apache-2.0). The Gated DeltaNet layer follows
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT). Reference tests
use [HuggingFace transformers](https://github.com/huggingface/transformers). Papers implemented
include DeepSeek-V2 (MLA), Mamba-2, Gated DeltaNet, [Kimi Linear](https://arxiv.org/abs/2510.26692) (KDA), Hyper-Connections,
[mHC](https://arxiv.org/abs/2512.24880) and [Attention Residuals](https://arxiv.org/abs/2603.15031).
See [NOTICE](NOTICE).

**Kimi Delta Attention.** `nn.KDA` implements KDA from Kimi Team,
[*Kimi Linear: An Expressive, Efficient Attention Architecture*](https://arxiv.org/abs/2510.26692)
(arXiv:2510.26692, 2025). The chunked algorithm is written from scratch in PyTorch to run on
Apple silicon (MPS), without the CUDA/Triton kernels of the official release: secondary decay
tiles keep the per-channel decay in float32 range, and the triangular inverse uses repeated
squaring instead of a triangular solve.

## Citation

```bibtex
@software{llm_blocks,
  title  = {llm-blocks: composable building blocks for LLM architecture research},
  url    = {https://github.com/cneuralnetwork/llm-blocks},
  year   = {2026}
}
```

If you use `nn.KDA`, please also cite the Kimi Linear paper:

```bibtex
@article{kimi_linear_2025,
  title   = {Kimi Linear: An Expressive, Efficient Attention Architecture},
  author  = {{Kimi Team}},
  journal = {arXiv preprint arXiv:2510.26692},
  year    = {2025}
}
```

## License

[Apache 2.0](LICENSE)
