import importlib
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from .utils import get_default_args

try:
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward

    HAS_FLASH_ATTN = True
except ImportError:
    _flash_attn_forward = None
    _flash_attn_backward = None
    HAS_FLASH_ATTN = False

_VALID_BACKENDS = {"auto", "flash_attn", "fused", "portable"}
_BACKEND_PRIORITY = ("flash_attn", "fused", "portable")
_BACKEND = os.environ.get("RINGX_ATTN_BACKEND", "auto")

_FUSED_MODULE = None
_FUSED_API = None
_FUSED_IMPORT_ERROR = None


@dataclass(frozen=True)
class _ForwardCall:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    softmax_scale: Optional[float]
    dropout_p: float = 0.0
    causal: bool = False
    window_size: Tuple[int, int] = (-1, -1)
    alibi_slopes: Optional[torch.Tensor] = None
    deterministic: bool = False


@dataclass(frozen=True)
class _BackwardCall:
    dout: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    softmax_lse: torch.Tensor
    softmax_scale: Optional[float]
    dropout_p: float = 0.0
    causal: bool = False
    window_size: Tuple[int, int] = (-1, -1)
    alibi_slopes: Optional[torch.Tensor] = None
    deterministic: bool = False


@dataclass(frozen=True)
class _BackendAdapter:
    name: str
    available: Callable[[], bool]
    unavailable_error: Optional[Callable[[], str]]
    forward: Callable[[_ForwardCall], Tuple[torch.Tensor, torch.Tensor]]
    backward: Callable[[_BackwardCall], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    forward_support_error: Optional[Callable[[_ForwardCall], Optional[str]]] = None
    backward_support_error: Optional[Callable[[_BackwardCall], Optional[str]]] = None


def _load_fused_module():
    global _FUSED_MODULE, _FUSED_API, _FUSED_IMPORT_ERROR
    if _FUSED_MODULE is not None:
        return _FUSED_MODULE
    if _FUSED_IMPORT_ERROR is not None:
        return None

    try:
        module = importlib.import_module("ringX_attn.fused_attention")
        get_api = getattr(module, "get_backend_api", None)
        if get_api is not None:
            api = get_api()
        else:
            api = getattr(module, "FUSED_BACKEND_API", None)
        if api is None:
            raise RuntimeError("the optional Triton fused attention module does not expose a backend API.")
        _FUSED_MODULE = module
        _FUSED_API = api
    except Exception as exc:
        _FUSED_IMPORT_ERROR = exc
        _FUSED_MODULE = None
        _FUSED_API = None
    return _FUSED_MODULE


def _load_fused_api():
    module = _load_fused_module()
    return None if module is None else _FUSED_API


def _fused_import_error_message() -> str:
    if _FUSED_IMPORT_ERROR is None:
        return "the optional Triton fused attention module is not available."
    return f"the optional Triton fused attention module could not be imported: {_FUSED_IMPORT_ERROR}"


def _fused_support_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0.0,
    window_size=(-1, -1),
    alibi_slopes=None,
) -> Optional[str]:
    api = _load_fused_api()
    if api is None:
        return _fused_import_error_message()
    return api.forward_support_error(
        q,
        k,
        v,
        dropout_p=dropout_p,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
    )


def _fused_backward_support_error(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dropout_p=0.0,
    window_size=(-1, -1),
    alibi_slopes=None,
) -> Optional[str]:
    api = _load_fused_api()
    if api is None:
        return _fused_import_error_message()
    return api.backward_support_error(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dropout_p=dropout_p,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
    )


def _requested_backend_name(backend: Optional[str]) -> str:
    return _BACKEND if backend is None else backend



def _resolve_runtime_backend(requested: str, adapter: _BackendAdapter, support_error: Optional[str]) -> _BackendAdapter:
    if support_error is None:
        return adapter
    if requested == "auto":
        return _BACKEND_ADAPTERS["portable"]
    raise RuntimeError(
        f"backend='{adapter.name}' was requested, but the current attention call is not supported: "
        f"{support_error} Use backend='portable', call set_backend('portable'), or set "
        "RINGX_ATTN_BACKEND=portable."
    )



def _forward_runtime_adapter(backend: Optional[str], call: _ForwardCall) -> _BackendAdapter:
    requested = _requested_backend_name(backend)
    adapter = _BACKEND_ADAPTERS[resolve_backend(backend)]
    support_error = None if adapter.forward_support_error is None else adapter.forward_support_error(call)
    return _resolve_runtime_backend(requested, adapter, support_error)



def _backward_runtime_adapter(backend: Optional[str], call: _BackwardCall) -> _BackendAdapter:
    requested = _requested_backend_name(backend)
    adapter = _BACKEND_ADAPTERS[resolve_backend(backend)]
    support_error = None if adapter.backward_support_error is None else adapter.backward_support_error(call)
    return _resolve_runtime_backend(requested, adapter, support_error)


def set_backend(backend: str) -> None:
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of {_VALID_BACKENDS}.")
    global _BACKEND
    _BACKEND = backend


def get_backend() -> str:
    return _BACKEND


def resolve_backend(backend: Optional[str] = None) -> str:
    selected = _requested_backend_name(backend)
    if selected not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported backend '{selected}'. Expected one of {_VALID_BACKENDS}.")
    if selected == "auto":
        for name in _BACKEND_PRIORITY:
            if _BACKEND_ADAPTERS[name].available():
                return name
        return "portable"

    adapter = _BACKEND_ADAPTERS[selected]
    if adapter.available():
        return selected

    unavailable_error = "backend is not available."
    if adapter.unavailable_error is not None:
        unavailable_error = adapter.unavailable_error()
    raise RuntimeError(
        f"backend='{selected}' was requested, but {unavailable_error} "
        "Use backend='portable', call set_backend('portable'), or set "
        "RINGX_ATTN_BACKEND=portable."
    )



def available_backends():
    return tuple(
        name
        for name in ("portable", "fused", "flash_attn")
        if _BACKEND_ADAPTERS[name].available()
    )



def forward_support_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: Optional[str] = None,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Optional[str]:
    selected = _requested_backend_name(backend)
    if selected not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported backend '{selected}'. Expected one of {_VALID_BACKENDS}.")
    if selected == "auto":
        return None

    adapter = _BACKEND_ADAPTERS[selected]
    if not adapter.available():
        return adapter.unavailable_error() if adapter.unavailable_error is not None else "backend is not available."

    if adapter.forward_support_error is None:
        return None

    call = _ForwardCall(
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )
    return adapter.forward_support_error(call)



def backward_support_error(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    *,
    backend: Optional[str] = None,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> Optional[str]:
    selected = _requested_backend_name(backend)
    if selected not in _VALID_BACKENDS:
        raise ValueError(f"Unsupported backend '{selected}'. Expected one of {_VALID_BACKENDS}.")
    if selected == "auto":
        return None

    adapter = _BACKEND_ADAPTERS[selected]
    if not adapter.available():
        return adapter.unavailable_error() if adapter.unavailable_error is not None else "backend is not available."

    if adapter.backward_support_error is None:
        return None

    call = _BackwardCall(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        softmax_lse=softmax_lse,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )
    return adapter.backward_support_error(call)



def runtime_forward_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    backend: Optional[str] = None,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> str:
    call = _ForwardCall(
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )
    return _forward_runtime_adapter(backend, call).name



def runtime_backward_backend(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    *,
    backend: Optional[str] = None,
    softmax_scale: Optional[float] = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> str:
    call = _BackwardCall(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        softmax_lse=softmax_lse,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )
    return _backward_runtime_adapter(backend, call).name


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


def _resolve_softmax_scale(softmax_scale: Optional[float], head_dim: int) -> float:
    return head_dim ** (-0.5) if softmax_scale is None else softmax_scale



def _to_head_first_layout(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1, 3).contiguous()



def _to_seq_first_layout(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1, 3).contiguous()


def _fused_forward(
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
    error = _fused_support_error(
        q,
        k,
        v,
        dropout_p=dropout_p,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
    )
    if error is not None:
        raise RuntimeError(f"Unable to run fused backend: {error}")

    softmax_scale = _resolve_softmax_scale(softmax_scale, q.shape[-1])

    fused_api = _load_fused_api()
    assert fused_api is not None

    qh = _to_head_first_layout(q)
    kh = _to_head_first_layout(k)
    vh = _to_head_first_layout(v)
    out, lse = fused_api.forward(qh, kh, vh, causal, softmax_scale)
    out = _to_seq_first_layout(out)
    return out.to(q.dtype), lse.contiguous()



def _fused_backward(
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
    error = _fused_backward_support_error(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dropout_p=dropout_p,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
    )
    if error is not None:
        raise RuntimeError(f"Unable to run fused backend: {error}")

    softmax_scale = _resolve_softmax_scale(softmax_scale, q.shape[-1])

    fused_api = _load_fused_api()
    assert fused_api is not None

    qh = _to_head_first_layout(q)
    kh = _to_head_first_layout(k)
    vh = _to_head_first_layout(v)
    douth = _to_head_first_layout(dout)
    outh = _to_head_first_layout(out)
    lse = softmax_lse.contiguous()

    dqh, dkh, dvh = fused_api.backward(
        qh,
        kh,
        vh,
        outh,
        lse,
        douth,
        causal=causal,
        sm_scale=softmax_scale,
    )
    dq = _to_seq_first_layout(dqh).to(q.dtype)
    dk = _to_seq_first_layout(dkh).to(k.dtype)
    dv = _to_seq_first_layout(dvh).to(v.dtype)
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


def _flash_is_available() -> bool:
    return HAS_FLASH_ATTN



def _flash_unavailable_error() -> str:
    return "flash_attn is not installed."



def _flash_adapter_forward(call: _ForwardCall):
    return _flash_forward(
        call.q,
        call.k,
        call.v,
        softmax_scale=call.softmax_scale,
        dropout_p=call.dropout_p,
        causal=call.causal,
        window_size=call.window_size,
        alibi_slopes=call.alibi_slopes,
        deterministic=call.deterministic,
    )



def _flash_adapter_backward(call: _BackwardCall):
    return _flash_backward(
        call.dout,
        call.q,
        call.k,
        call.v,
        call.out,
        call.softmax_lse,
        softmax_scale=call.softmax_scale,
        dropout_p=call.dropout_p,
        causal=call.causal,
        window_size=call.window_size,
        alibi_slopes=call.alibi_slopes,
        deterministic=call.deterministic,
    )



def _fused_is_available() -> bool:
    return _load_fused_api() is not None



def _fused_unavailable_error() -> str:
    return f"{_fused_import_error_message()} Install Triton support for this environment, or use backend='portable'."



def _fused_adapter_forward_support_error(call: _ForwardCall) -> Optional[str]:
    return _fused_support_error(
        call.q,
        call.k,
        call.v,
        dropout_p=call.dropout_p,
        window_size=call.window_size,
        alibi_slopes=call.alibi_slopes,
    )



def _fused_adapter_backward_support_error(call: _BackwardCall) -> Optional[str]:
    return _fused_backward_support_error(
        call.dout,
        call.q,
        call.k,
        call.v,
        call.out,
        call.softmax_lse,
        dropout_p=call.dropout_p,
        window_size=call.window_size,
        alibi_slopes=call.alibi_slopes,
    )



def _fused_adapter_forward(call: _ForwardCall):
    return _fused_forward(
        call.q,
        call.k,
        call.v,
        softmax_scale=call.softmax_scale,
        dropout_p=call.dropout_p,
        causal=call.causal,
        window_size=call.window_size,
        alibi_slopes=call.alibi_slopes,
        deterministic=call.deterministic,
    )



def _fused_adapter_backward(call: _BackwardCall):
    return _fused_backward(
        call.dout,
        call.q,
        call.k,
        call.v,
        call.out,
        call.softmax_lse,
        softmax_scale=call.softmax_scale,
        dropout_p=call.dropout_p,
        causal=call.causal,
        window_size=call.window_size,
        alibi_slopes=call.alibi_slopes,
        deterministic=call.deterministic,
    )



def _portable_is_available() -> bool:
    return True



def _portable_adapter_forward(call: _ForwardCall):
    return _portable_forward(
        call.q,
        call.k,
        call.v,
        softmax_scale=call.softmax_scale,
        dropout_p=call.dropout_p,
        causal=call.causal,
        window_size=call.window_size,
        alibi_slopes=call.alibi_slopes,
        deterministic=call.deterministic,
    )



def _portable_adapter_backward(call: _BackwardCall):
    return _portable_backward(
        call.dout,
        call.q,
        call.k,
        call.v,
        call.out,
        call.softmax_lse,
        softmax_scale=call.softmax_scale,
        dropout_p=call.dropout_p,
        causal=call.causal,
        window_size=call.window_size,
        alibi_slopes=call.alibi_slopes,
        deterministic=call.deterministic,
    )



_BACKEND_ADAPTERS: Dict[str, _BackendAdapter] = {
    "flash_attn": _BackendAdapter(
        name="flash_attn",
        available=_flash_is_available,
        unavailable_error=_flash_unavailable_error,
        forward=_flash_adapter_forward,
        backward=_flash_adapter_backward,
    ),
    "fused": _BackendAdapter(
        name="fused",
        available=_fused_is_available,
        unavailable_error=_fused_unavailable_error,
        forward=_fused_adapter_forward,
        backward=_fused_adapter_backward,
        forward_support_error=_fused_adapter_forward_support_error,
        backward_support_error=_fused_adapter_backward_support_error,
    ),
    "portable": _BackendAdapter(
        name="portable",
        available=_portable_is_available,
        unavailable_error=None,
        forward=_portable_adapter_forward,
        backward=_portable_adapter_backward,
    ),
}



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
    call = _ForwardCall(
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )
    return _forward_runtime_adapter(backend, call).forward(call)



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
    call = _BackwardCall(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        softmax_lse=softmax_lse,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )
    return _backward_runtime_adapter(backend, call).backward(call)
