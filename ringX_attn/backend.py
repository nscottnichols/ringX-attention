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


def _portable_q_tile_size(batch: int, nheads: int, seqlen_k: int, device: torch.device) -> int:
    override = os.environ.get("RINGX_ATTN_PORTABLE_Q_TILE")
    if override is not None:
        q_tile = int(override)
        if q_tile <= 0:
            raise ValueError("Portable Q tile size must be a positive integer.")
        return q_tile

    target_mb = int(os.environ.get("RINGX_ATTN_PORTABLE_SCORE_CHUNK_MB", "256"))
    if target_mb <= 0:
        raise ValueError("RINGX_ATTN_PORTABLE_SCORE_CHUNK_MB must be a positive integer.")

    bytes_per_score = torch.tensor([], device=device, dtype=torch.float32).element_size()
    denom = max(batch * nheads * seqlen_k * bytes_per_score, 1)
    q_tile = max(16, (target_mb * 1024 * 1024) // denom)
    return max(16, q_tile)



def _build_block_mask(
    q_start: int,
    q_end: int,
    seq_k: int,
    causal: bool,
    window_size: Tuple[int, int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    invalid = None
    if causal or window_size[0] >= 0 or window_size[1] >= 0:
        q_pos = torch.arange(q_start, q_end, device=device).unsqueeze(-1)
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

    qh = q.permute(0, 2, 1, 3).contiguous().to(torch.float32)
    kh = k.permute(0, 2, 1, 3).contiguous().to(torch.float32)
    vh = v.permute(0, 2, 1, 3).contiguous().to(torch.float32)

    batch, nheads, seqlen_q, head_dim = qh.shape
    seqlen_k = kh.shape[-2]
    q_tile = min(seqlen_q, _portable_q_tile_size(batch, nheads, seqlen_k, q.device))

    batch_heads = batch * nheads
    qh = qh.reshape(batch_heads, seqlen_q, head_dim)
    kh = kh.reshape(batch_heads, seqlen_k, head_dim)
    vh = vh.reshape(batch_heads, seqlen_k, head_dim)
    kh_t = kh.transpose(-1, -2).contiguous()

    out = torch.empty((batch_heads, seqlen_q, head_dim), device=q.device, dtype=torch.float32)
    lse = torch.empty((batch_heads, seqlen_q), device=q.device, dtype=torch.float32)

    for q_start in range(0, seqlen_q, q_tile):
        q_end = min(q_start + q_tile, seqlen_q)
        scores = torch.bmm(qh[:, q_start:q_end, :], kh_t) * softmax_scale

        invalid = _build_block_mask(q_start, q_end, seqlen_k, causal, window_size, q.device)
        if invalid is not None:
            scores.masked_fill_(invalid.unsqueeze(0), float("-inf"))

        block_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.where(
            torch.isfinite(block_lse).unsqueeze(-1),
            torch.exp(scores - block_lse.unsqueeze(-1)),
            torch.zeros_like(scores),
        )

        out[:, q_start:q_end, :] = torch.bmm(probs, vh)
        lse[:, q_start:q_end] = block_lse

    out = out.reshape(batch, nheads, seqlen_q, head_dim).permute(0, 2, 1, 3).contiguous()
    lse = lse.reshape(batch, nheads, seqlen_q).contiguous()
    return out.to(q.dtype), lse



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

    qh = q.permute(0, 2, 1, 3).contiguous().to(torch.float32)
    kh = k.permute(0, 2, 1, 3).contiguous().to(torch.float32)
    vh = v.permute(0, 2, 1, 3).contiguous().to(torch.float32)
    douth = dout.permute(0, 2, 1, 3).contiguous().to(torch.float32)
    outh = out.permute(0, 2, 1, 3).contiguous().to(torch.float32)
    lse = softmax_lse.contiguous().to(torch.float32)

    batch, nheads, seqlen_q, head_dim = qh.shape
    seqlen_k = kh.shape[-2]
    q_tile = min(seqlen_q, _portable_q_tile_size(batch, nheads, seqlen_k, q.device))

    batch_heads = batch * nheads
    qh = qh.reshape(batch_heads, seqlen_q, head_dim)
    kh = kh.reshape(batch_heads, seqlen_k, head_dim)
    vh = vh.reshape(batch_heads, seqlen_k, head_dim)
    douth = douth.reshape(batch_heads, seqlen_q, head_dim)
    outh = outh.reshape(batch_heads, seqlen_q, head_dim)
    lse = lse.reshape(batch_heads, seqlen_q)

    kh_t = kh.transpose(-1, -2).contiguous()
    vh_t = vh.transpose(-1, -2).contiguous()

    dq = torch.empty_like(qh)
    dk = torch.zeros_like(kh)
    dv = torch.zeros_like(vh)
    delta = (douth * outh).sum(dim=-1)

    for q_start in range(0, seqlen_q, q_tile):
        q_end = min(q_start + q_tile, seqlen_q)
        q_blk = qh[:, q_start:q_end, :]
        do_blk = douth[:, q_start:q_end, :]
        lse_blk = lse[:, q_start:q_end]
        delta_blk = delta[:, q_start:q_end]

        scores = torch.bmm(q_blk, kh_t) * softmax_scale
        invalid = _build_block_mask(q_start, q_end, seqlen_k, causal, window_size, q.device)
        if invalid is not None:
            scores.masked_fill_(invalid.unsqueeze(0), float("-inf"))

        p = torch.where(
            torch.isfinite(lse_blk).unsqueeze(-1),
            torch.exp(scores - lse_blk.unsqueeze(-1)),
            torch.zeros_like(scores),
        )
        dp = torch.bmm(do_blk, vh_t)
        ds = p * (dp - delta_blk.unsqueeze(-1))

        dq[:, q_start:q_end, :] = torch.bmm(ds, kh) * softmax_scale
        dk += torch.bmm(ds.transpose(-1, -2), q_blk) * softmax_scale
        dv += torch.bmm(p.transpose(-1, -2), do_blk)

    dq = dq.reshape(batch, nheads, seqlen_q, head_dim).permute(0, 2, 1, 3).contiguous().to(q.dtype)
    dk = dk.reshape(batch, nheads, seqlen_k, head_dim).permute(0, 2, 1, 3).contiguous().to(k.dtype)
    dv = dv.reshape(batch, nheads, seqlen_k, head_dim).permute(0, 2, 1, 3).contiguous().to(v.dtype)
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
