import math
import os
from typing import Optional, Tuple

import torch

from .utils import get_default_args

try:
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    HAS_FLASH_ATTN = True
except ImportError:
    _flash_attn_forward = None
    _flash_attn_backward = None
    HAS_FLASH_ATTN = False

_VALID_BACKENDS = {"auto", "flash_attn", "portable"}
_BACKEND = os.environ.get("RINGX_ATTN_BACKEND", "auto")


def set_backend(backend: str) -> None:
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of {_VALID_BACKENDS}.")
    global _BACKEND
    _BACKEND = backend


def get_backend() -> str:
    return _BACKEND


def resolve_backend(backend: Optional[str] = None) -> str:
    selected = _BACKEND if backend is None else backend
    if selected not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported backend '{selected}'. Expected one of {_VALID_BACKENDS}.")
    if selected == "auto":
        return "flash_attn" if HAS_FLASH_ATTN else "portable"
    if selected == "flash_attn" and not HAS_FLASH_ATTN:
        raise RuntimeError(
            "backend='flash_attn' was requested, but flash_attn is not installed. "
            "Use backend='portable', call set_backend('portable'), or set "
            "RINGX_ATTN_BACKEND=portable."
        )
    return selected


def available_backends():
    backends = ["portable"]
    if HAS_FLASH_ATTN:
        backends.append("flash_attn")
    return tuple(backends)


def _build_local_mask(
    seq_q: int,
    seq_k: int,
    causal: bool,
    window_size: Tuple[int, int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    invalid = None
    q_pos = torch.arange(seq_q, device=device).unsqueeze(-1)
    k_pos = torch.arange(seq_k, device=device).unsqueeze(0)

    if causal:
        invalid = k_pos > q_pos

    window_left, window_right = window_size
    if window_left >= 0:
        left_invalid = k_pos < (q_pos - window_left)
        invalid = left_invalid if invalid is None else (invalid | left_invalid)
    if window_right >= 0:
        right_invalid = k_pos > (q_pos + window_right)
        invalid = right_invalid if invalid is None else (invalid | right_invalid)

    return invalid


def _flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    params = get_default_args(_flash_attn_forward).copy()
    if "window_size" in params:
        params.update({"window_size": window_size})
    else:
        params.update(
            {
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
            }
        )
    params.update(
        {
            "q": q,
            "k": k,
            "v": v,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "alibi_slopes": alibi_slopes,
            "return_softmax": True and dropout_p > 0,
        }
    )
    outputs = _flash_attn_forward(**params)
    if len(outputs) == 8:
        out, _, _, _, _, lse, _, _ = outputs
    else:
        assert len(outputs) == 4
        out, lse, _, _ = outputs
    return out, lse


def _flash_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    seqlen_q = q.shape[1]
    seqlen_kv = k.shape[1]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    params = get_default_args(_flash_attn_backward).copy()
    if "window_size" in params:
        params.update({"window_size": window_size})
    else:
        params.update(
            {
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
            }
        )
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    params.update(
        {
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": softmax_lse,
            "dq": dq[:, :seqlen_q],
            "dk": dk[:, :seqlen_kv],
            "dv": dv[:, :seqlen_kv],
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "alibi_slopes": alibi_slopes,
            "deterministic": deterministic,
            "rng_state": rng_state,
        }
    )
    _flash_attn_backward(**params)
    return dq, dk, dv


def _portable_tile_sizes() -> Tuple[int, int]:
    q_tile = int(os.environ.get("RINGX_ATTN_PORTABLE_Q_TILE", "128"))
    k_tile = int(os.environ.get("RINGX_ATTN_PORTABLE_K_TILE", "64"))
    if q_tile <= 0 or k_tile <= 0:
        raise ValueError("Portable tile sizes must be positive integers.")
    return q_tile, k_tile



def _apply_local_mask_(
    scores: torch.Tensor,
    q_start: int,
    k_start: int,
    causal: bool,
    window_size: Tuple[int, int],
) -> None:
    invalid = None
    q_len = scores.shape[-2]
    k_len = scores.shape[-1]
    if causal or window_size[0] >= 0 or window_size[1] >= 0:
        q_pos = torch.arange(q_start, q_start + q_len, device=scores.device).unsqueeze(-1)
        k_pos = torch.arange(k_start, k_start + k_len, device=scores.device).unsqueeze(0)
        if causal:
            invalid = k_pos > q_pos
        window_left, window_right = window_size
        if window_left >= 0:
            left_invalid = k_pos < (q_pos - window_left)
            invalid = left_invalid if invalid is None else (invalid | left_invalid)
        if window_right >= 0:
            right_invalid = k_pos > (q_pos + window_right)
            invalid = right_invalid if invalid is None else (invalid | right_invalid)
    if invalid is not None:
        scores.masked_fill_(invalid.unsqueeze(0).unsqueeze(0), float("-inf"))



def _portable_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    if dropout_p != 0:
        raise NotImplementedError("The portable backend currently supports dropout_p=0 only.")
    if alibi_slopes is not None:
        raise NotImplementedError("The portable backend does not support alibi_slopes.")

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    qh = q.permute(0, 2, 1, 3).to(torch.float32)
    kh = k.permute(0, 2, 1, 3).to(torch.float32)
    vh = v.permute(0, 2, 1, 3).to(torch.float32)

    batch, nheads, seqlen_q, head_dim = qh.shape
    seqlen_k = kh.shape[-2]
    q_tile, k_tile = _portable_tile_sizes()

    out = torch.empty((batch, nheads, seqlen_q, head_dim), device=q.device, dtype=torch.float32)
    lse = torch.empty((batch, nheads, seqlen_q), device=q.device, dtype=torch.float32)

    for q_start in range(0, seqlen_q, q_tile):
        q_end = min(q_start + q_tile, seqlen_q)
        q_blk = qh[:, :, q_start:q_end, :]
        q_len = q_end - q_start

        m_i = torch.full((batch, nheads, q_len), float("-inf"), device=q.device, dtype=torch.float32)
        l_i = torch.zeros((batch, nheads, q_len), device=q.device, dtype=torch.float32)
        acc = torch.zeros((batch, nheads, q_len, head_dim), device=q.device, dtype=torch.float32)

        for k_start in range(0, seqlen_k, k_tile):
            k_end = min(k_start + k_tile, seqlen_k)
            k_blk = kh[:, :, k_start:k_end, :]
            v_blk = vh[:, :, k_start:k_end, :]

            scores = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * softmax_scale
            _apply_local_mask_(scores, q_start, k_start, causal, window_size)

            block_m = scores.amax(dim=-1)
            new_m = torch.maximum(m_i, block_m)
            alpha = torch.where(
                torch.isfinite(m_i),
                torch.exp(m_i - new_m),
                torch.zeros_like(new_m),
            )
            p = torch.where(
                torch.isfinite(scores),
                torch.exp(scores - new_m.unsqueeze(-1)),
                torch.zeros_like(scores),
            )
            l_i = l_i * alpha + p.sum(dim=-1)
            acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, v_blk)
            m_i = new_m

        valid = l_i > 0
        out[:, :, q_start:q_end, :] = torch.where(
            valid.unsqueeze(-1),
            acc / l_i.unsqueeze(-1).clamp_min(1e-20),
            torch.zeros_like(acc),
        )
        lse[:, :, q_start:q_end] = torch.where(
            valid,
            torch.log(l_i) + m_i,
            torch.full_like(m_i, float("-inf")),
        )

    out = out.permute(0, 2, 1, 3).contiguous()
    return out.to(q.dtype), lse.contiguous()



def _portable_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    if dropout_p != 0:
        raise NotImplementedError("The portable backend currently supports dropout_p=0 only.")
    if alibi_slopes is not None:
        raise NotImplementedError("The portable backend does not support alibi_slopes.")

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    qh = q.permute(0, 2, 1, 3).to(torch.float32)
    kh = k.permute(0, 2, 1, 3).to(torch.float32)
    vh = v.permute(0, 2, 1, 3).to(torch.float32)
    douth = dout.permute(0, 2, 1, 3).to(torch.float32)
    outh = out.permute(0, 2, 1, 3).to(torch.float32)
    lse = softmax_lse.to(torch.float32)

    batch, nheads, seqlen_q, head_dim = qh.shape
    seqlen_k = kh.shape[-2]
    q_tile, k_tile = _portable_tile_sizes()

    dq = torch.zeros_like(qh)
    dk = torch.zeros_like(kh)
    dv = torch.zeros_like(vh)
    delta = (douth * outh).sum(dim=-1)

    for q_start in range(0, seqlen_q, q_tile):
        q_end = min(q_start + q_tile, seqlen_q)
        q_blk = qh[:, :, q_start:q_end, :]
        do_blk = douth[:, :, q_start:q_end, :]
        lse_blk = lse[:, :, q_start:q_end]
        delta_blk = delta[:, :, q_start:q_end]
        dq_blk = torch.zeros_like(q_blk)

        for k_start in range(0, seqlen_k, k_tile):
            k_end = min(k_start + k_tile, seqlen_k)
            k_blk = kh[:, :, k_start:k_end, :]
            v_blk = vh[:, :, k_start:k_end, :]

            scores = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * softmax_scale
            _apply_local_mask_(scores, q_start, k_start, causal, window_size)

            p = torch.where(
                torch.isfinite(scores) & torch.isfinite(lse_blk.unsqueeze(-1)),
                torch.exp(scores - lse_blk.unsqueeze(-1)),
                torch.zeros_like(scores),
            )
            dp = torch.matmul(do_blk, v_blk.transpose(-1, -2))
            ds = p * (dp - delta_blk.unsqueeze(-1))

            dq_blk = dq_blk + torch.matmul(ds, k_blk) * softmax_scale
            dk[:, :, k_start:k_end, :] = dk[:, :, k_start:k_end, :] + torch.matmul(ds.transpose(-1, -2), q_blk) * softmax_scale
            dv[:, :, k_start:k_end, :] = dv[:, :, k_start:k_end, :] + torch.matmul(p.transpose(-1, -2), do_blk)

        dq[:, :, q_start:q_end, :] = dq_blk

    dq = dq.permute(0, 2, 1, 3).contiguous().to(q.dtype)
    dk = dk.permute(0, 2, 1, 3).contiguous().to(k.dtype)
    dv = dv.permute(0, 2, 1, 3).contiguous().to(v.dtype)
    return dq, dk, dv


def local_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    backend: Optional[str] = None,
):
    selected = resolve_backend(backend)
    if selected == "flash_attn":
        return _flash_forward(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
    return _portable_forward(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )


def local_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_scale,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    backend: Optional[str] = None,
):
    selected = resolve_backend(backend)
    if selected == "flash_attn":
        return _flash_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
    return _portable_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )
