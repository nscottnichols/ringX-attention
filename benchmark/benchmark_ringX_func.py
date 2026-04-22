import json
import os
import traceback
from datetime import timedelta

import argparse
import importlib
import torch
import torch.distributed as dist

from ringX_attn import backend as ringx_backend

try:
    from ring_flash_attn import (
        ring_flash_attn_func,
        zigzag_ring_flash_attn_func,
        stripe_flash_attn_func,
    )
    baseline_funcs = [
        zigzag_ring_flash_attn_func,
        stripe_flash_attn_func,
        ring_flash_attn_func,
    ]
except ModuleNotFoundError:
    baseline_funcs = []


def shard_simple(x, rank, world_size):
    return x.chunk(world_size, dim=1)[rank].contiguous()


def shard_balanced(x, rank, world_size):
    parts = x.chunk(2 * world_size, dim=1)
    return torch.cat([parts[rank], parts[2 * world_size - rank - 1]], dim=1).contiguous()


def _is_ringx_impl(func):
    return getattr(func, "__module__", "").startswith("ringX_attn.")



def _current_backend_name(func):
    return ringx_backend.get_backend() if _is_ringx_impl(func) else "external"


_FUSED_BENCHMARK_UNSUPPORTED_ALGOS = {
    "ringX3_attn": "ringX3_attn slices q and kv into unequal local sequence blocks, which the fused backend does not support.",
    "ringX4_attn": "ringX4_attn slices q and kv into unequal local sequence blocks, which the fused backend does not support.",
    "ringX4o_attn": "ringX4o_attn slices q and kv into unequal local sequence blocks, which the fused backend does not support.",
}



def _algo_backend_support_error(args, requested_backend):
    if requested_backend != "fused":
        return None
    algo = args.module.rsplit(".", 1)[-1]
    reason = _FUSED_BENCHMARK_UNSUPPORTED_ALGOS.get(algo)
    if reason is None:
        return None
    return (
        f"benchmark preflight: backend='fused' is not supported for {algo}. {reason} "
        "Use backend='portable', backend='auto', or benchmark an algorithm whose local attention calls keep q, k, and v aligned."
    )



def _preflight_result(args, func, q, k, v, dout, *, causal, mode, deterministic=False):
    if not _is_ringx_impl(func):
        return {
            "status": "ready",
            "requested_backend": "external",
            "forward_backend": "external",
            "backward_backend": "external",
            "reason": "",
        }

    requested_backend = ringx_backend.get_backend()
    algo_error = _algo_backend_support_error(args, requested_backend)
    if algo_error is not None:
        return {
            "status": "skipped",
            "requested_backend": requested_backend,
            "forward_backend": "unsupported",
            "backward_backend": "unsupported" if mode != "forward" else "n/a",
            "reason": algo_error,
        }

    kwargs = dict(
        dropout_p=0.0,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
    )
    forward_error = ringx_backend.forward_support_error(
        q,
        k,
        v,
        backend=requested_backend,
        **kwargs,
    )
    if forward_error is not None:
        return {
            "status": "skipped",
            "requested_backend": requested_backend,
            "forward_backend": "unsupported",
            "backward_backend": "unsupported" if mode != "forward" else "n/a",
            "reason": forward_error,
        }

    forward_backend = ringx_backend.runtime_forward_backend(
        q,
        k,
        v,
        backend=requested_backend,
        **kwargs,
    )

    backward_backend = "n/a"
    if mode != "forward":
        out_probe = torch.empty_like(q)
        lse_probe = torch.empty(
            q.shape[0],
            q.shape[2],
            q.shape[1],
            device=q.device,
            dtype=torch.float32,
        )
        backward_error = ringx_backend.backward_support_error(
            dout,
            q,
            k,
            v,
            out_probe,
            lse_probe,
            backend=requested_backend,
            **kwargs,
        )
        if backward_error is not None:
            return {
                "status": "skipped",
                "requested_backend": requested_backend,
                "forward_backend": forward_backend,
                "backward_backend": "unsupported",
                "reason": backward_error,
            }
        backward_backend = ringx_backend.runtime_backward_backend(
            dout,
            q,
            k,
            v,
            out_probe,
            lse_probe,
            backend=requested_backend,
            **kwargs,
        )

    return {
        "status": "ready",
        "requested_backend": requested_backend,
        "forward_backend": forward_backend,
        "backward_backend": backward_backend,
        "reason": "",
    }



def _collect_preflight(preflight):
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, preflight)
    for entry in gathered:
        if entry is None:
            continue
        if entry.get("status") != "ready":
            return entry
    return preflight



def _zero_grads(*tensors):
    for tensor in tensors:
        tensor.grad = None



def _run_forward(func, q, k, v, causal, deterministic):
    return func(
        q,
        k,
        v,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=False,
        group=dist.group.WORLD,
    )



def _warmup(mode, func, q, k, v, dout, *, warmup_iter, causal, deterministic):
    for _ in range(warmup_iter):
        _zero_grads(q, k, v)
        if mode == "forward":
            with torch.no_grad():
                _ = _run_forward(func, q, k, v, causal, deterministic)
            continue

        out = _run_forward(func, q, k, v, causal, deterministic)
        out.backward(dout)



def _measure_forward(func, q, k, v, *, num_iter, causal, deterministic, profile=False, profiler=None):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    with torch.no_grad():
        for _ in range(num_iter):
            _ = _run_forward(func, q, k, v, causal, deterministic)
            if profile and profiler is not None:
                profiler.step()
    end.record()
    torch.cuda.synchronize(device=q.device)
    return begin.elapsed_time(end) / 1000.0



def _measure_forward_backward(func, q, k, v, dout, *, num_iter, causal, deterministic, profile=False, profiler=None):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(num_iter):
        _zero_grads(q, k, v)
        out = _run_forward(func, q, k, v, causal, deterministic)
        out.backward(dout)
        if profile and profiler is not None:
            profiler.step()
    end.record()
    torch.cuda.synchronize(device=q.device)
    return begin.elapsed_time(end) / 1000.0



def _measure_backward(func, q, k, v, dout, *, num_iter, causal, deterministic, profile=False, profiler=None):
    total_ms = 0.0
    for _ in range(num_iter):
        _zero_grads(q, k, v)
        out = _run_forward(func, q, k, v, causal, deterministic)
        torch.cuda.synchronize(device=q.device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        out.backward(dout)
        end.record()
        torch.cuda.synchronize(device=q.device)
        total_ms += begin.elapsed_time(end)
        if profile and profiler is not None:
            profiler.step()
    return total_ms / 1000.0



def benchmark(args, func, warmup_iter=1, num_iter=100, mode="forward", log=True, profile=False):
    dtype = getattr(torch, args.dtype)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    batch_size = args.batch_size
    deterministic = False
    seqlen = args.seq_length
    num_heads = args.num_heads
    head_dim = args.head_dim
    causal = args.causal

    assert seqlen % (2 * world_size) == 0
    assert head_dim % 8 == 0

    if rank == 0:
        print(
            f"ngpus: {world_size}, causal: {causal}, batch: {batch_size}, seqlen: {seqlen}, "
            f"num_heads: {num_heads}, head_dim: {head_dim}"
        )

    q = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    dout = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )

    algo = args.module.rsplit(".", 1)[-1]
    if algo in {"ringX1_attn", "ringX2_attn"}:
        q = shard_simple(q, rank, world_size)
        k = shard_simple(k, rank, world_size)
        v = shard_simple(v, rank, world_size)
        dout = shard_simple(dout, rank, world_size)
    elif algo in {"ringX3_attn", "ringX3b_attn", "ringX4_attn", "ringX4o_attn"}:
        q = shard_balanced(q, rank, world_size)
        k = shard_balanced(k, rank, world_size)
        v = shard_balanced(v, rank, world_size)
        dout = shard_balanced(dout, rank, world_size)

    preflight = _collect_preflight(_preflight_result(
        args,
        func,
        q,
        k,
        v,
        dout,
        causal=causal,
        mode=mode,
        deterministic=deterministic,
    ))
    if preflight["status"] != "ready":
        return {
            "status": preflight["status"],
            "requested_backend": preflight["requested_backend"],
            "forward_backend": preflight["forward_backend"],
            "backward_backend": preflight["backward_backend"],
            "iter_per_s": None,
            "total_sec": None,
            "reason": preflight["reason"],
        }

    profiler = None
    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=5),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(f"./benchmark/logs/{func.__name__}", f"rank_{dist.get_rank()}")
            ),
        )

    _warmup(
        mode,
        func,
        q,
        k,
        v,
        dout,
        warmup_iter=warmup_iter,
        causal=causal,
        deterministic=deterministic,
    )

    if profile and profiler is not None:
        profiler.start()

    if mode == "forward":
        total_sec = _measure_forward(
            func,
            q,
            k,
            v,
            num_iter=num_iter,
            causal=causal,
            deterministic=deterministic,
            profile=profile,
            profiler=profiler,
        )
    elif mode == "fwd_bwd":
        total_sec = _measure_forward_backward(
            func,
            q,
            k,
            v,
            dout,
            num_iter=num_iter,
            causal=causal,
            deterministic=deterministic,
            profile=profile,
            profiler=profiler,
        )
    elif mode == "backward":
        total_sec = _measure_backward(
            func,
            q,
            k,
            v,
            dout,
            num_iter=num_iter,
            causal=causal,
            deterministic=deterministic,
            profile=profile,
            profiler=profiler,
        )
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    if profile and profiler is not None:
        profiler.stop()

    iter_per_s = num_iter / total_sec if total_sec > 0 else float("inf")
    if rank == 0 and log:
        print(f"{iter_per_s:.6f} iter/s, {total_sec:.3f} sec")

    return {
        "status": "ok",
        "requested_backend": preflight["requested_backend"],
        "forward_backend": preflight["forward_backend"],
        "backward_backend": preflight["backward_backend"],
        "iter_per_s": iter_per_s,
        "total_sec": total_sec,
        "reason": "",
    }



def _emit_result(args, func, mode, result):
    if dist.get_rank() != 0:
        return
    payload = {
        "algo": args.module,
        "impl": func.__name__,
        "mode": mode,
        "status": result["status"],
        "requested_backend": result["requested_backend"],
        "forward_backend": result["forward_backend"],
        "backward_backend": result["backward_backend"],
        "causal": args.causal,
        "batch": args.batch_size,
        "seqlen": args.seq_length,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "ngpus": dist.get_world_size(),
        "iter_per_s": result["iter_per_s"],
        "total_sec": result["total_sec"],
        "reason": result["reason"],
    }
    print("BENCHMARK_RESULT " + json.dumps(payload, sort_keys=True))



def _resolve_modes(args):
    if args.modes:
        return args.modes
    if args.forward_only:
        return ["forward"]
    return ["fwd_bwd"]



def _available_impls(main_func):
    return {func.__name__: func for func in [main_func, *baseline_funcs]}



def _resolve_impls(args, main_func):
    impls = _available_impls(main_func)
    if args.impl is None:
        return list(impls.values())
    try:
        return [impls[args.impl]]
    except KeyError as exc:
        names = ", ".join(sorted(impls))
        raise ValueError(f"Unknown impl '{args.impl}'. Available implementations: {names}") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse model configuration arguments.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training or inference.")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for input data.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--module", type=str, required=True, help="Module name to import the function.")
    parser.add_argument("--impl", type=str, help="Specific implementation name to benchmark. Defaults to the module implementation plus any available baselines.")
    parser.add_argument("--causal", action="store_true", help="Enable causal attention masking.")
    parser.add_argument("--forward_only", action="store_true", help="Legacy alias for --modes forward.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("forward", "backward", "fwd_bwd"),
        help="Benchmark modes to run. Defaults to forward when --forward_only is set, otherwise fwd_bwd.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default=os.environ.get("BENCHMARK_DTYPE", "bfloat16"),
        help="Tensor dtype to benchmark.",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling.")
    dist.init_process_group("nccl", timeout=timedelta(seconds=36000))
    rank = dist.get_rank()
    args = parser.parse_args()

    try:
        module = importlib.import_module(args.module)
        func_name = f"{args.module.rsplit('.', 1)[-1]}_func"
        ringX_attn_func = getattr(module, func_name)
    except ModuleNotFoundError:
        if rank == 0:
            print(f"Error: Module '{args.module}' not found.")
        dist.destroy_process_group()
        raise

    if rank == 0:
        print(f"Algo: {args.module}")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.seq_length}")
        print(f"Number of heads: {args.num_heads}")
        print(f"Head dimension: {args.head_dim}")
        print(f"Dtype: {args.dtype}")

    profile = args.profile
    num_iter = args.num_iter
    modes = _resolve_modes(args)
    impls = _resolve_impls(args, ringX_attn_func)
    exit_code = 0

    try:
        for mode in modes:
            for func in impls:
                torch.cuda.empty_cache()
                if rank == 0:
                    print(f"# {func.__name__} [{mode}]")
                try:
                    result = benchmark(
                        args,
                        func,
                        mode=mode,
                        num_iter=num_iter,
                        log=True,
                        profile=profile,
                    )
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "requested_backend": _current_backend_name(func),
                        "forward_backend": "error",
                        "backward_backend": "error" if mode != "forward" else "n/a",
                        "iter_per_s": None,
                        "total_sec": None,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                    exit_code = 1
                    if rank == 0:
                        print(f"Benchmark failed for {func.__name__} [{mode}]: {exc}")
                        print(traceback.format_exc())
                _emit_result(args, func, mode, result)
                if result["status"] == "failed":
                    raise RuntimeError(result["reason"])
    finally:
        dist.destroy_process_group()

    raise SystemExit(exit_code)
