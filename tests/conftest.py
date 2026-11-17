import pytest
import torch

from llm_blocks import nn

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def tiny_mixers():
    """One small instance of every sequence mixer, keyed by name."""
    return {
        "MHA": nn.MHA(heads=4),
        "GQA": nn.GQA(heads=4, kv_heads=2),
        "MQA": nn.MQA(heads=4),
        "GQA-qknorm": nn.GQA(heads=4, kv_heads=2, qk_norm=True),
        "GQA-alibi": nn.GQA(heads=4, kv_heads=2, position=nn.ALiBi()),
        "GQA-nope": nn.GQA(heads=4, kv_heads=2, position=None),
        "MLA": nn.MLA(heads=4),
        "MLA-qlora": nn.MLA(heads=4, q_lora_rank=32, kv_lora_rank=24, qk_rope_head_dim=8),
        "Mamba2": nn.Mamba2(d_state=16, headdim=16, chunk_size=8),
        "Mamba2-groups": nn.Mamba2(d_state=8, headdim=16, ngroups=2, chunk_size=4),
        "GatedDeltaNet": nn.GatedDeltaNet(heads=2, head_dim=16, chunk_size=8),
        "KDA": nn.KDA(heads=2, head_dim=16, decay_rank=8, chunk_size=8),
    }


def tiny_model(mixer, ffn=None, depth=2, dim=64, vocab=97, **kw) -> nn.LLM:
    cfg = nn.LLMConfig(vocab_size=vocab, dim=dim, depth=depth, mixer=mixer,
                       ffn=ffn if ffn is not None else nn.SwiGLU(hidden_dim=96), max_seq_len=128, **kw)
    return nn.LLM(config=cfg)
