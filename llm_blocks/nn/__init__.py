"""``llm_blocks.nn``: the building-block namespace.

Everything a user composes is available here::

    from llm_blocks import nn

    model = nn.LLM(
        vocab_size=32000, dim=768,
        layers=[nn.Block(mixer=nn.GQA(heads=12, kv_heads=4, position=nn.RoPE()),
                         ffn=nn.SwiGLU(hidden_dim=2048), norm=nn.RMSNorm()) for _ in range(12)],
    )

Names like ``nn.GQA`` / ``nn.SwiGLU`` are *specs* (serializable descriptions). The
``nn.Module`` implementations live in the submodules (``llm_blocks.nn.attention`` etc.) and
are also usable directly.
"""

from ..configs.spec import BuildContext, Spec, SpecError, spec_from_dict
from .attention import GQA, MHA, MLA, MQA, Attention, KVCache, MLACache, MultiHeadLatentAttention
from .base import FeedForward, FFNSpec, ForwardContext, MixerSpec, MixerState, SequenceMixer
from .blocks import Block, BlockModule
from .ffn import FFN, GLU, GeGLU, GeluFFN, ReluFFN, SwiGLU
from .mixers import KDA, GatedDeltaNet, GatedDeltaNetMixer, KimiDeltaAttention, Mamba2, Mamba2Mixer
from .models import LLM, LLMConfig, LMOutput, ModelState, every
from .moe import MoE, MoELayer
from .norm import LayerNorm, NormSpec, QKNorm, RMSNorm
from .position import ALiBi, LearnedPositions, NoPE, PositionEncoding, PositionSpec, RoPE
from .residual import AttnRes, GatedResidual, HyperConnections, MHC, Residual, ResidualSpec, ResidualStrategy

__all__ = [
    # model
    "LLM", "LLMConfig", "LMOutput", "ModelState", "Block", "BlockModule", "every",
    # mixers
    "MHA", "MQA", "GQA", "MLA", "Mamba2", "GatedDeltaNet", "KDA",
    # feed-forward
    "FFN", "GeluFFN", "ReluFFN", "GLU", "SwiGLU", "GeGLU", "MoE",
    # normalization
    "RMSNorm", "LayerNorm", "QKNorm",
    # positions
    "RoPE", "ALiBi", "LearnedPositions", "NoPE",
    # residuals
    "Residual", "GatedResidual", "HyperConnections", "MHC", "AttnRes",
    # interfaces
    "SequenceMixer", "FeedForward", "ResidualStrategy", "PositionEncoding", "MixerState", "ForwardContext",
    "Spec", "MixerSpec", "FFNSpec", "NormSpec", "PositionSpec", "ResidualSpec", "BuildContext", "SpecError",
    "spec_from_dict",
    # implementations
    "Attention", "KVCache", "MultiHeadLatentAttention", "MLACache", "Mamba2Mixer", "GatedDeltaNetMixer", "KimiDeltaAttention", "MoELayer",
]
