import torch
import torch.distributed as dist

def sync_scalar_across_ranks(value, device="cuda"):
    tensor = torch.tensor([value], dtype=torch.int, device=device)
    dist.broadcast(tensor, src=0)
    return tensor.item()