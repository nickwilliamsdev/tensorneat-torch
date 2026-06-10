import torch


def sum_(z, mask):
    return torch.sum(torch.where(mask, z, torch.zeros_like(z)), dim=0)


def product_(z, mask):
    return torch.prod(torch.where(mask, z, torch.ones_like(z)), dim=0)


def max_(z, mask):
    return torch.max(torch.where(mask, z, torch.full_like(z, -torch.inf)), dim=0).values


def min_(z, mask):
    return torch.min(torch.where(mask, z, torch.full_like(z, torch.inf)), dim=0).values


def maxabs_(z, mask):
    masked = torch.where(mask, z, torch.zeros_like(z))
    index = torch.argmax(torch.abs(masked), dim=0, keepdim=True)
    gathered = torch.take_along_dim(masked, index, dim=0)
    return gathered.squeeze(0)


def mean_(z, mask):
    total = torch.sum(torch.where(mask, z, torch.zeros_like(z)), dim=0)
    count = torch.sum(mask, dim=0)
    safe_count = torch.clamp(count, min=1)
    return total / safe_count.to(dtype=z.dtype)