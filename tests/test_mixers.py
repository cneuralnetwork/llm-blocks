"""Behavioural tests shared by every sequence mixer."""

import pytest
import torch

from llm_blocks import nn
from llm_blocks.nn import BuildContext

from conftest import DEVICES, tiny_mixers

MIXERS = tiny_mixers()
DIM = 64


def build(name, device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    m = MIXERS[name].build(BuildContext(dim=DIM, max_seq_len=128)).to(device=device, dtype=dtype)
    return m.eval()


@pytest.mark.parametrize("name", list(MIXERS))
def test_shape_and_backward(name):
    m = build(name).train()
    x = torch.randn(2, 13, DIM, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for n, p in m.named_parameters():
        assert p.grad is not None, f"{name}: no grad for {n}"
        assert torch.isfinite(p.grad).all(), n


@pytest.mark.parametrize("name", list(MIXERS))
def test_causality(name):
    m = build(name)
    x = torch.randn(1, 20, DIM)
    x2 = x.clone()
    x2[:, 12:] = torch.randn(1, 8, DIM)
    with torch.no_grad():
        y, y2 = m(x), m(x2)
    torch.testing.assert_close(y[:, :12], y2[:, :12], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(y[:, 12:], y2[:, 12:])


@pytest.mark.parametrize("name", list(MIXERS))
@pytest.mark.parametrize("split", [[23], [1] * 23, [7, 1, 1, 9, 5], [8, 8, 7]])
def test_incremental_matches_full(name, split):
    """Feeding the sequence in chunks through the mixer's own state == full causal pass."""
    m = build(name)
    x = torch.randn(2, sum(split), DIM)
    with torch.no_grad():
        full = m(x)
        state = m.init_state(2, 32)
        outs, t = [], 0
        for n in split:
            outs.append(m(x[:, t:t + n], state=state))
            t += n
    torch.testing.assert_close(torch.cat(outs, 1), full, atol=2e-5, rtol=2e-5)
    assert state.offset == x.shape[1]


@pytest.mark.parametrize("name", list(MIXERS))
def test_left_padding_is_exact(name):
    """A left-padded sequence gives the same outputs (on real tokens) as the unpadded one."""
    m = build(name)
    x = torch.randn(1, 10, DIM)
    pad = torch.randn(1, 5, DIM) * 3
    xp = torch.cat([pad, x], 1)
    mask = torch.cat([torch.zeros(1, 5), torch.ones(1, 10)], 1).bool()
    with torch.no_grad():
        ref = m(x)
        out = m(xp, padding_mask=mask)[:, 5:]
        # also through a decoding state: padded prefill, then decode
        state = m.init_state(1, 32)
        pre = m(xp[:, :12], state=state, padding_mask=mask[:, :12])[:, 5:]
        rest = [m(xp[:, t:t + 1], state=state) for t in range(12, 15)]
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.cat([pre] + rest, 1), ref, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("name", list(MIXERS))
def test_right_padding_does_not_affect_real_tokens(name):
    m = build(name)
    x = torch.randn(1, 10, DIM)
    xp = torch.cat([x, torch.randn(1, 4, DIM)], 1)
    mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 4)], 1).bool()
    with torch.no_grad():
        torch.testing.assert_close(m(xp, padding_mask=mask)[:, :10], m(x), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("name", list(MIXERS))
@pytest.mark.parametrize("device", DEVICES)
def test_bf16_close_to_fp32(name, device):
    m32 = build(name, device)
    m16 = build(name, device, torch.bfloat16)
    x = torch.randn(2, 16, DIM, device=device)
    with torch.no_grad():
        y32 = m32(x)
        y16 = m16(x.bfloat16())
    assert y16.dtype == torch.bfloat16 and torch.isfinite(y16).all()
    rel = (y16.float() - y32).norm() / y32.norm()
    assert rel < 0.05, rel


@pytest.mark.parametrize("name", list(MIXERS))
@pytest.mark.skipif(len(DEVICES) == 1, reason="no accelerator")
def test_accelerator_matches_cpu(name):
    dev = DEVICES[1]
    mc, md = build(name), build(name, dev)
    x = torch.randn(2, 17, DIM)
    with torch.no_grad():
        torch.testing.assert_close(md(x.to(dev)).cpu(), mc(x), atol=1e-4, rtol=1e-4)
        st = md.init_state(2, 32)
        md(x[:, :16].to(dev), state=st)
        last = md(x[:, 16:].to(dev), state=st).cpu()
    torch.testing.assert_close(last, mc(x)[:, 16:], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name", ["GQA", "GQA-alibi", "GQA-nope", "MLA"])
def test_sdpa_matches_eager_backend(name):
    spec = MIXERS[name].replace(backend="eager")
    torch.manual_seed(0)
    eager = spec.build(BuildContext(dim=DIM)).eval()
    sdpa = build(name)
    x = torch.randn(2, 11, DIM)
    mask = torch.ones(2, 11, dtype=torch.bool)
    mask[0, :3] = False
    with torch.no_grad():
        torch.testing.assert_close(sdpa(x), eager(x), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sdpa(x, padding_mask=mask)[:, 3:], eager(x, padding_mask=mask)[:, 3:], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", ["GQA", "Mamba2", "GatedDeltaNet", "KDA", "MLA"])
def test_gradcheck_float64(name):
    spec = MIXERS[name]
    if hasattr(spec, "backend"):
        spec = spec.replace(backend="eager")
    torch.manual_seed(0)
    m = spec.build(BuildContext(dim=16 if name != "MLA" else 16)).double()
    x = torch.randn(1, 5, 16, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: m(t), (x,), eps=1e-6, atol=1e-5)


def test_mla_absorbed_equals_expanded_decode():
    m = build("MLA-qlora")
    m2 = build("MLA-qlora")
    m2.absorb_for_decode = False
    x = torch.randn(2, 12, DIM)
    with torch.no_grad():
        s1, s2 = m.init_state(2, 16), m2.init_state(2, 16)
        a = torch.cat([m(x[:, :6], state=s1)] + [m(x[:, t:t + 1], state=s1) for t in range(6, 12)], 1)
        b = torch.cat([m2(x[:, :6], state=s2)] + [m2(x[:, t:t + 1], state=s2) for t in range(6, 12)], 1)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_cache_sizes():
    gqa, mla = build("GQA"), build("MLA")
    s1, s2 = gqa.init_state(1, 100), mla.init_state(1, 100)
    # GQA caches 2 * kv_heads * head_dim per token; MLA caches kv_lora_rank + rope dim
    assert s1.k.numel() + s1.v.numel() == 100 * 2 * 2 * 16
    assert s2.c_kv.numel() + s2.k_pe.numel() == 100 * (mla.kv_lora_rank + mla.rope)
    mamba = build("Mamba2")
    s3 = mamba.init_state(1, 100_000)  # recurrent state does not grow with length
    assert s3.nbytes() < 50_000


def test_kv_cache_overflow_raises():
    m = build("GQA")
    st = m.init_state(1, 4)
    with torch.no_grad():
        m(torch.randn(1, 4, DIM), state=st)
        with pytest.raises(RuntimeError, match="overflow"):
            m(torch.randn(1, 1, DIM), state=st)


def test_numerical_stability_large_inputs():
    for name in MIXERS:
        m = build(name)
        with torch.no_grad():
            y = m(torch.randn(1, 40, DIM) * 1e3)
        assert torch.isfinite(y).all(), name
