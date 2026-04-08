import random
import torch
import torch.distributed as dist


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(name, value, rank0_only=False):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank0_only and rank != 0:
        return

    if torch.is_tensor(value):
        x = value.detach().float()
        max_abs = x.abs().max().item() if x.numel() else 0.0
        mean_abs = x.abs().mean().item() if x.numel() else 0.0
        print(
            f"[rank {rank}] {name}: shape={list(x.shape)} "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}",
            flush=True,
        )
    else:
        print(f"[rank {rank}] {name}: {value}", flush=True)
