import contextlib
import os

import torch
import torch.distributed as dist

import ringX_attn.backend as backend


@contextlib.contextmanager
def patched_attrs(module, **updates):
    originals = {}
    try:
        for name, value in updates.items():
            originals[name] = getattr(module, name)
            setattr(module, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def _sample_tensors():
    q = torch.randn(1, 128, 2, 64)
    k = torch.randn(1, 128, 2, 64)
    v = torch.randn(1, 128, 2, 64)
    return q, k, v


def _init_dist_if_needed():
    if dist.is_available() and not dist.is_initialized() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("gloo")
        return True
    return False


def _rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def test_auto_uses_fused_when_available_and_supported():
    q, k, v = _sample_tensors()
    fused_out = torch.full_like(q, 7.0)
    fused_lse = torch.full((q.shape[0], q.shape[2], q.shape[1]), 3.0)

    def _portable_should_not_run(*args, **kwargs):
        raise AssertionError("portable backend should not have been selected")

    with patched_attrs(
        backend,
        HAS_FLASH_ATTN=False,
        _load_fused_attn=lambda: object(),
        _fused_support_error=lambda *args, **kwargs: None,
        _fused_forward=lambda *args, **kwargs: (fused_out, fused_lse),
        _portable_forward=_portable_should_not_run,
    ):
        out, lse = backend.local_attn_forward(q, k, v, softmax_scale=None, backend="auto")
        assert out is fused_out
        assert lse is fused_lse



def test_auto_falls_back_to_portable_when_fused_is_unsupported():
    q, k, v = _sample_tensors()
    portable_out = torch.full_like(q, 11.0)
    portable_lse = torch.full((q.shape[0], q.shape[2], q.shape[1]), 5.0)

    def _fused_should_not_run(*args, **kwargs):
        raise AssertionError("fused backend should not have been selected")

    with patched_attrs(
        backend,
        HAS_FLASH_ATTN=False,
        _load_fused_attn=lambda: object(),
        _fused_support_error=lambda *args, **kwargs: "fused backend does not support local window attention.",
        _fused_forward=_fused_should_not_run,
        _portable_forward=lambda *args, **kwargs: (portable_out, portable_lse),
    ):
        out, lse = backend.local_attn_forward(q, k, v, softmax_scale=None, backend="auto")
        assert out is portable_out
        assert lse is portable_lse



def test_explicit_fused_raises_for_unsupported_call():
    q, k, v = _sample_tensors()

    with patched_attrs(
        backend,
        _load_fused_attn=lambda: object(),
        _fused_support_error=lambda *args, **kwargs: "fused backend does not support local window attention.",
    ):
        try:
            backend.local_attn_forward(q, k, v, softmax_scale=None, backend="fused")
        except RuntimeError as exc:
            assert "backend='fused'" in str(exc)
        else:
            raise AssertionError("Expected backend='fused' unsupported call to raise RuntimeError")


if __name__ == "__main__":
    initialized = _init_dist_if_needed()
    rank = _rank()

    test_auto_uses_fused_when_available_and_supported()
    if rank == 0:
        print("[ok] auto selects fused when available and supported", flush=True)

    test_auto_falls_back_to_portable_when_fused_is_unsupported()
    if rank == 0:
        print("[ok] auto falls back to portable when fused is unsupported", flush=True)

    test_explicit_fused_raises_for_unsupported_call()
    if rank == 0:
        print("[ok] explicit fused backend raises for unsupported calls", flush=True)

    _barrier()
    if initialized:
        dist.destroy_process_group()

