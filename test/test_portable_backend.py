import torch

from ringX_attn.backend import local_attn_backward, local_attn_forward


def _naive_attention(q, k, v, softmax_scale, causal=False):
    qh = q.permute(0, 2, 1, 3).float()
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * softmax_scale
    if causal:
        mask = torch.triu(torch.ones(q.shape[1], k.shape[1], device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh).permute(0, 2, 1, 3).contiguous()
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


def test_portable_forward_backward_matches_naive():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 3, 8, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 4, 3, 8, dtype=torch.float32, requires_grad=True)
    v = torch.randn(2, 4, 3, 8, dtype=torch.float32, requires_grad=True)
    scale = q.shape[-1] ** -0.5

    ref_out, ref_lse = _naive_attention(q, k, v, scale, causal=True)
    dout = torch.randn_like(ref_out)
    ref_out.backward(dout)
    ref_dq = q.grad.clone()
    ref_dk = k.grad.clone()
    ref_dv = v.grad.clone()

    with torch.no_grad():
        out, lse = local_attn_forward(q.detach(), k.detach(), v.detach(), scale, causal=True, backend="portable")
        dq, dk, dv = local_attn_backward(dout, q.detach(), k.detach(), v.detach(), out, lse, scale, causal=True, backend="portable")

    assert torch.allclose(out.float(), ref_out.detach().float(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(lse.float(), ref_lse.detach().float(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(dq.float(), ref_dq.float(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(dk.float(), ref_dk.float(), atol=1e-5, rtol=1e-5)
    assert torch.allclose(dv.float(), ref_dv.float(), atol=1e-5, rtol=1e-5)
