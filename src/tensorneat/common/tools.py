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
    flat_mask = mask.reshape(-1)
    default_tensor = torch.as_tensor(default, dtype=torch.int64, device=mask.device)
    if flat_mask.numel() == 0:
        return default_tensor

    idx = torch.argmax(flat_mask.to(dtype=torch.int64))
    return torch.where(flat_mask[idx], idx.to(dtype=torch.int64), default_tensor)


def fetch_random(generator, mask, default=I_INF):
    flat_mask = mask.reshape(-1)
    default_tensor = torch.as_tensor(default, dtype=torch.int64, device=mask.device)
    if flat_mask.numel() == 0:
        return default_tensor

    flat_mask_int = flat_mask.to(dtype=torch.int64)
    true_cnt = torch.sum(flat_mask_int)
    cumsum = torch.cumsum(flat_mask_int, dim=0)
    target = (
        torch.floor(
            torch.rand((), generator=generator, device=mask.device, dtype=torch.float32)
            * true_cnt.to(dtype=torch.float32)
        ).to(dtype=torch.int64)
        + 1
    )
    target_mask = torch.where(true_cnt == 0, torch.zeros_like(flat_mask), cumsum >= target)
    return fetch_first(target_mask, default_tensor)


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