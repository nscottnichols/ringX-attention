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


_FUSED_UNSUPPORTED_PREFIX = "backend='fused' was requested, but the current attention call is not supported:"


def _classify_benchmark_exception(func, mode, exc):
    requested_backend = _current_backend_name(func)
    reason = f"{type(exc).__name__}: {exc}"
    if requested_backend == "fused" and _FUSED_UNSUPPORTED_PREFIX in str(exc):
        return {
            "status": "skipped",
            "requested_backend": requested_backend,
            "forward_backend": "unsupported",
            "backward_backend": "unsupported" if mode != "forward" else "n/a",
            "iter_per_s": None,
            "total_sec": None,
            "reason": reason,
        }
    return {
        "status": "failed",
        "requested_backend": requested_backend,
        "forward_backend": "error",
        "backward_backend": "error" if mode != "forward" else "n/a",
        "iter_per_s": None,
        "total_sec": None,
        "reason": reason,
    }


def _merge_rank_results(results):
    statuses = {result["status"] for result in results}
    if statuses == {"ok"}:
        return results[0]

    if len(statuses) > 1:
        rank_summaries = [f"rank{rank}:{result['status']}" for rank, result in enumerate(results)]
        first_non_ok = next(result for result in results if result["status"] != "ok")
        merged = dict(first_non_ok)
        merged.update(
            status="failed",
            iter_per_s=None,
            total_sec=None,
            forward_backend=first_non_ok.get("forward_backend", "error"),
            backward_backend=first_non_ok.get("backward_backend", "n/a"),
            reason=(
                "inconsistent benchmark outcome across distributed ranks: "
                + ", ".join(rank_summaries)
                + f"; first non-ok reason: {first_non_ok['reason']}"
            ),
        )
        return merged

    if statuses == {"skipped"}:
        return results[0]

    return next(result for result in results if result["status"] == "failed")



def _run_benchmark_distributed(args, func, mode, *, num_iter, log, profile):
    rank = dist.get_rank()
    local_traceback = ""
    try:
        local_result = benchmark(
            args,
            func,
            mode=mode,
            num_iter=num_iter,
            log=log,
            profile=profile,
        )
    except Exception as exc:
        local_result = _classify_benchmark_exception(func, mode, exc)
        if local_result["status"] == "failed":
            local_traceback = traceback.format_exc()

    gathered_results = [None for _ in range(dist.get_world_size())]
    gathered_tracebacks = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_results, local_result)
    dist.all_gather_object(gathered_tracebacks, local_traceback)
    merged_result = _merge_rank_results(gathered_results)

    if rank == 0:
        if merged_result["status"] == "failed":
            print(f"Benchmark failed for {func.__name__} [{mode}]: {merged_result['reason']}")
            for failed_rank, (result, tb) in enumerate(zip(gathered_results, gathered_tracebacks)):
                if result["status"] != "failed":
                    continue
                print(f"[rank{failed_rank}] {result['reason']}")
                if tb:
                    print(tb)
        elif merged_result["status"] == "skipped":
            print(f"Benchmark skipped for {func.__name__} [{mode}]: {merged_result['reason']}")
    return merged_result



def _preflight_result(func, q, k, v, dout, *, causal, mode, deterministic=False):
    if not _is_ringx_impl(func):
        return {
            "status": "ready",
            "requested_backend": "external",
            "forward_backend": "external",
            "backward_backend": "external",
            "reason": "",
        }

    kwargs = dict(
        dropout_p=0.0,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
    )
    requested_backend = ringx_backend.get_backend()
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

    preflight = _preflight_result(
        func,
        q,
        k,
        v,
        dout,
        causal=causal,
        mode=mode,
        deterministic=deterministic,
    )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse model configuration arguments.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training or inference.")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for input data.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--module", type=str, required=True, help="Module name to import the function.")
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

    for mode in modes:
        for func in [ringX_attn_func, *baseline_funcs]:
            torch.cuda.empty_cache()
            if rank == 0:
                print(f"# {func.__name__} [{mode}]")
            result = _run_benchmark_distributed(
                args,
                func,
                mode,
                num_iter=num_iter,
                log=True,
                profile=profile,
            )
            _emit_result(args, func, mode, result)

    dist.destroy_process_group()
