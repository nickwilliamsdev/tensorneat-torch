import zlib

import torch

# Infinite int used to represent an unavailable index in integer tensors.
I_INF = torch.iinfo(torch.int32).max


def attach_with_inf(arr, idx):
    safe_idx = torch.where(idx == I_INF, torch.zeros_like(idx), idx)
    gathered = arr[safe_idx]

    missing_mask = idx == I_INF
    while missing_mask.ndim < gathered.ndim:
        missing_mask = missing_mask.unsqueeze(-1)

    return torch.where(missing_mask, torch.full_like(gathered, torch.nan), gathered)


def fetch_first(mask, default=I_INF):
    indices = torch.nonzero(mask, as_tuple=False)
    if indices.numel() == 0:
        return torch.tensor(default, dtype=torch.int64, device=mask.device)
    return indices[0, 0].to(dtype=torch.int64)


def fetch_random(generator, mask, default=I_INF):
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return torch.tensor(default, dtype=torch.int64, device=mask.device)

    choice = torch.randint(
        low=0,
        high=indices.numel(),
        size=(),
        generator=generator,
        device=mask.device,
    )
    return indices[choice].to(dtype=torch.int64)


def rank_elements(array, reverse=False):
    values = array if reverse else -array
    first_sort = torch.argsort(values)
    return torch.argsort(first_sort)


def mutate_float(
    generator,
    val,
    init_mean,
    init_std,
    mutate_power,
    mutate_rate,
    replace_rate,
):
    noise = torch.randn((), generator=generator, device=val.device, dtype=val.dtype) * mutate_power
    replace = (
        torch.randn((), generator=generator, device=val.device, dtype=val.dtype) * init_std
        + init_mean
    )
    rand = torch.rand((), generator=generator, device=val.device, dtype=val.dtype)

    mutated = val + noise
    replaced = torch.where(
        (mutate_rate < rand) & (rand < mutate_rate + replace_rate),
        replace,
        val,
    )
    return torch.where(rand < mutate_rate, mutated, replaced)


def mutate_int(generator, val, options, replace_rate):
    rand = torch.rand((), generator=generator, device=val.device)
    replace_idx = torch.randint(
        low=0,
        high=options.numel(),
        size=(),
        generator=generator,
        device=options.device,
    )
    replaced = options.reshape(-1)[replace_idx].to(dtype=val.dtype, device=val.device)
    return torch.where(rand < replace_rate, replaced, val)


def argmin_with_mask(arr, mask):
    masked_arr = torch.where(mask, arr, torch.full_like(arr, torch.inf))
    return torch.argmin(masked_arr)


def hash_array(arr):
    buffer = arr.detach().contiguous().cpu().numpy().tobytes()
    return zlib.crc32(buffer)