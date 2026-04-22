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


def test_forward_dispatch_uses_backend_adapter_table():
    q, k, v = _sample_tensors()
    captured = {}
    adapter_out = torch.full_like(q, 29.0)
    adapter_lse = torch.full((q.shape[0], q.shape[2], q.shape[1]), 31.0)

    def _forward(call):
        captured["call_type"] = type(call).__name__
        captured["shapes"] = (call.q.shape, call.k.shape, call.v.shape)
        captured["window_size"] = call.window_size
        captured["deterministic"] = call.deterministic
        return adapter_out, adapter_lse

    adapters = dict(backend._BACKEND_ADAPTERS)
    adapters["portable"] = backend._BackendAdapter(
        name="portable",
        available=lambda: True,
        unavailable_error=None,
        forward=_forward,
        backward=lambda call: (_ for _ in ()).throw(AssertionError("backward should not be called")),
    )

    with patched_attrs(backend, _BACKEND_ADAPTERS=adapters):
        out, lse = backend.local_attn_forward(
            q,
            k,
            v,
            softmax_scale=None,
            window_size=(7, 9),
            deterministic=True,
            backend="portable",
        )

    assert out is adapter_out
    assert lse is adapter_lse
    assert captured["call_type"] == "_ForwardCall"
    assert captured["shapes"] == (q.shape, k.shape, v.shape)
    assert captured["window_size"] == (7, 9)
    assert captured["deterministic"] is True



def test_backward_dispatch_uses_backend_adapter_table():
    q, k, v = _sample_tensors()
    dout = torch.randn_like(q)
    out = torch.randn_like(q)
    lse = torch.randn(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)
    captured = {}
    grads = tuple(torch.full_like(tensor, fill_value) for tensor, fill_value in ((q, 37.0), (k, 41.0), (v, 43.0)))

    def _backward(call):
        captured["call_type"] = type(call).__name__
        captured["dout_shape"] = call.dout.shape
        captured["out_shape"] = call.out.shape
        captured["lse_shape"] = call.softmax_lse.shape
        captured["causal"] = call.causal
        captured["softmax_scale"] = call.softmax_scale
        return grads

    adapters = dict(backend._BACKEND_ADAPTERS)
    adapters["portable"] = backend._BackendAdapter(
        name="portable",
        available=lambda: True,
        unavailable_error=None,
        forward=lambda call: (_ for _ in ()).throw(AssertionError("forward should not be called")),
        backward=_backward,
    )

    with patched_attrs(backend, _BACKEND_ADAPTERS=adapters):
        dq, dk, dv = backend.local_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            softmax_scale=0.25,
            causal=True,
            backend="portable",
        )

    assert dq is grads[0]
    assert dk is grads[1]
    assert dv is grads[2]
    assert captured["call_type"] == "_BackwardCall"
    assert captured["dout_shape"] == dout.shape
    assert captured["out_shape"] == out.shape
    assert captured["lse_shape"] == lse.shape
    assert captured["causal"] is True
    assert captured["softmax_scale"] == 0.25


def test_auto_uses_fused_when_available_and_supported():
    q, k, v = _sample_tensors()
    fused_out = torch.full_like(q, 7.0)
    fused_lse = torch.full((q.shape[0], q.shape[2], q.shape[1]), 3.0)

    def _portable_should_not_run(*args, **kwargs):
        raise AssertionError("portable backend should not have been selected")

    with patched_attrs(
        backend,
        HAS_FLASH_ATTN=False,
        _load_fused_api=lambda: object(),
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
        _load_fused_api=lambda: object(),
        _fused_support_error=lambda *args, **kwargs: "fused backend does not support local window attention.",
        _fused_forward=_fused_should_not_run,
        _portable_forward=lambda *args, **kwargs: (portable_out, portable_lse),
    ):
        out, lse = backend.local_attn_forward(q, k, v, softmax_scale=None, backend="auto")
        assert out is portable_out
        assert lse is portable_lse



def test_backend_delegates_support_checks_to_fused_module():
    q, k, v = _sample_tensors()
    portable_out = torch.full_like(q, 13.0)
    portable_lse = torch.full((q.shape[0], q.shape[2], q.shape[1]), 6.0)
    calls = {}

    class _FakeFusedAPI:
        def forward_support_error(self, q_, k_, v_, dropout_p=0.0, window_size=(-1, -1), alibi_slopes=None):
            calls["shapes"] = (q_.shape, k_.shape, v_.shape)
            calls["dtype"] = q_.dtype
            calls["device_type"] = q_.device.type
            calls["dropout_p"] = dropout_p
            calls["window_size"] = window_size
            calls["alibi_slopes"] = alibi_slopes
            return "fused backend does not support local window attention."

    with patched_attrs(
        backend,
        HAS_FLASH_ATTN=False,
        _load_fused_api=lambda: _FakeFusedAPI(),
        _portable_forward=lambda *args, **kwargs: (portable_out, portable_lse),
    ):
        out, lse = backend.local_attn_forward(
            q,
            k,
            v,
            softmax_scale=None,
            window_size=(32, 32),
            backend="auto",
        )

    assert out is portable_out
    assert lse is portable_lse
    assert calls["shapes"][0] == q.shape
    assert calls["dtype"] == q.dtype
    assert calls["device_type"] == q.device.type
    assert calls["dropout_p"] == 0.0
    assert calls["window_size"] == (32, 32)
    assert calls["alibi_slopes"] is None


def test_explicit_fused_raises_for_unsupported_call():
    q, k, v = _sample_tensors()

    with patched_attrs(
        backend,
        _load_fused_api=lambda: object(),
        _fused_support_error=lambda *args, **kwargs: "fused backend does not support local window attention.",
    ):
        try:
            backend.local_attn_forward(q, k, v, softmax_scale=None, backend="fused")
        except RuntimeError as exc:
            assert "backend='fused'" in str(exc)
        else:
            raise AssertionError("Expected backend='fused' unsupported call to raise RuntimeError")


def test_fused_backward_uses_public_wrapper():
    q, k, v = _sample_tensors()
    dout = torch.randn_like(q)
    out = torch.randn_like(q)
    lse = torch.randn(q.shape[0], q.shape[2], q.shape[1])

    calls = {}

    class _FakeFusedAPI:
        def backward(self, qh, kh, vh, outh, lseh, douth, causal, sm_scale):
            calls["shapes"] = (qh.shape, kh.shape, vh.shape, outh.shape, lseh.shape, douth.shape)
            calls["causal"] = causal
            calls["sm_scale"] = sm_scale
            return qh + 1, kh + 2, vh + 3

    with patched_attrs(
        backend,
        _load_fused_api=lambda: _FakeFusedAPI(),
        _fused_backward_support_error=lambda *args, **kwargs: None,
    ):
        dq, dk, dv = backend.local_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            softmax_scale=0.125,
            causal=True,
            backend="fused",
        )

    assert calls["shapes"][0] == (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
    assert calls["shapes"][4] == lse.shape
    assert calls["causal"] is True
    assert calls["sm_scale"] == 0.125
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape


def test_backend_delegates_backward_support_checks_to_fused_module():
    q, k, v = _sample_tensors()
    dout = torch.randn_like(q)
    out = torch.randn_like(q)
    lse = torch.randn(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)
    portable_grads = tuple(torch.full_like(tensor, fill_value) for tensor, fill_value in ((q, 17.0), (k, 19.0), (v, 23.0)))
    calls = {}

    class _FakeFusedAPI:
        def backward_support_error(
            self,
            dout_,
            q_,
            k_,
            v_,
            out_,
            softmax_lse_,
            dropout_p=0.0,
            window_size=(-1, -1),
            alibi_slopes=None,
        ):
            calls["dout_shape"] = dout_.shape
            calls["out_shape"] = out_.shape
            calls["lse_shape"] = softmax_lse_.shape
            calls["lse_dtype"] = softmax_lse_.dtype
            calls["dropout_p"] = dropout_p
            calls["window_size"] = window_size
            calls["alibi_slopes"] = alibi_slopes
            return "fused backend requires softmax_lse to have shape (1, 2, 128)."

    with patched_attrs(
        backend,
        HAS_FLASH_ATTN=False,
        _load_fused_api=lambda: _FakeFusedAPI(),
        _portable_backward=lambda *args, **kwargs: portable_grads,
    ):
        dq, dk, dv = backend.local_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            softmax_scale=None,
            window_size=(32, 32),
            backend="auto",
        )

    assert dq is portable_grads[0]
    assert dk is portable_grads[1]
    assert dv is portable_grads[2]
    assert calls["dout_shape"] == dout.shape
    assert calls["out_shape"] == out.shape
    assert calls["lse_shape"] == lse.shape
    assert calls["lse_dtype"] == lse.dtype
    assert calls["dropout_p"] == 0.0
    assert calls["window_size"] == (32, 32)
    assert calls["alibi_slopes"] is None



def test_explicit_fused_backward_raises_for_unsupported_call():
    q, k, v = _sample_tensors()
    dout = torch.randn_like(q)
    out = torch.randn_like(q)
    lse = torch.randn(q.shape[0], q.shape[2], q.shape[1])

    with patched_attrs(
        backend,
        _load_fused_api=lambda: object(),
        _fused_backward_support_error=lambda *args, **kwargs: "fused backend currently supports dropout_p=0 only.",
    ):
        try:
            backend.local_attn_backward(
                dout,
                q,
                k,
                v,
                out,
                lse,
                softmax_scale=None,
                dropout_p=0.1,
                backend="fused",
            )
        except RuntimeError as exc:
            assert "backend='fused'" in str(exc)
        else:
            raise AssertionError("Expected backend='fused' unsupported backward call to raise RuntimeError")


if __name__ == "__main__":
    initialized = _init_dist_if_needed()
    rank = _rank()

    test_forward_dispatch_uses_backend_adapter_table()
    if rank == 0:
        print("[ok] forward dispatch uses the backend adapter table", flush=True)

    test_backward_dispatch_uses_backend_adapter_table()
    if rank == 0:
        print("[ok] backward dispatch uses the backend adapter table", flush=True)

    test_auto_uses_fused_when_available_and_supported()
    if rank == 0:
        print("[ok] auto selects fused when available and supported", flush=True)

    test_auto_falls_back_to_portable_when_fused_is_unsupported()
    if rank == 0:
        print("[ok] auto falls back to portable when fused is unsupported", flush=True)

    test_backend_delegates_support_checks_to_fused_module()
    if rank == 0:
        print("[ok] backend delegates fused support checks to the fused module", flush=True)

    test_explicit_fused_raises_for_unsupported_call()
    if rank == 0:
        print("[ok] explicit fused backend raises for unsupported calls", flush=True)

    test_fused_backward_uses_public_wrapper()
    if rank == 0:
        print("[ok] fused backward uses the public fused backend API wrapper", flush=True)

    test_backend_delegates_backward_support_checks_to_fused_module()
    if rank == 0:
        print("[ok] backend delegates fused backward support checks to the fused module", flush=True)

    test_explicit_fused_backward_raises_for_unsupported_call()
    if rank == 0:
        print("[ok] explicit fused backward raises for unsupported calls", flush=True)

    _barrier()
    if initialized:
        dist.destroy_process_group()

