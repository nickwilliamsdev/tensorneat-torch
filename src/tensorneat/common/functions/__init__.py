import torch

from .act_sympy import *
from .act_torch import *
from .agg_sympy import *
from .agg_torch import *
from .manager import FunctionManager

act_name2torch = {
    "scaled_sigmoid": scaled_sigmoid_,
    "sigmoid": sigmoid_,
    "scaled_tanh": scaled_tanh_,
    "tanh": tanh_,
    "sin": sin_,
    "relu": relu_,
    "lelu": lelu_,
    "identity": identity_,
    "inv": inv_,
    "log": log_,
    "exp": exp_,
    "abs": abs_,
}

act_name2sympy = {
    "scaled_sigmoid": SympyScaledSigmoid,
    "sigmoid": SympySigmoid,
    "scaled_tanh": SympyScaledTanh,
    "tanh": SympyTanh,
    "sin": SympySin,
    "relu": SympyRelu,
    "lelu": SympyLelu,
    "identity": SympyIdentity,
    "inv": SympyInv,
    "log": SympyLog,
    "exp": SympyExp,
    "abs": SympyAbs,
    "clip": SympyClip,
}

agg_name2torch = {
    "sum": sum_,
    "product": product_,
    "max": max_,
    "min": min_,
    "maxabs": maxabs_,
    "mean": mean_,
}

agg_name2sympy = {
    "sum": SympySum,
    "product": SympyProduct,
    "max": SympyMax,
    "min": SympyMin,
    "maxabs": SympyMaxabs,
    "mean": SympyMean,
}

ACT = FunctionManager(act_name2torch, act_name2sympy)
AGG = FunctionManager(agg_name2torch, agg_name2sympy)


def _coerce_index(idx, device=None):
    if isinstance(idx, torch.Tensor):
        if idx.ndim != 0:
            raise ValueError(f"Expected a scalar index tensor, got shape {tuple(idx.shape)}")
        return idx.to(device=device, dtype=torch.long)
    return torch.tensor(idx, device=device, dtype=torch.long)


def _switch(idx, funcs, *operands):
    if len(funcs) == 1:
        return funcs[0](*operands)

    split = len(funcs) // 2
    left_funcs = funcs[:split]
    right_funcs = funcs[split:]

    return torch.cond(
        idx < split,
        lambda: _switch(idx, left_funcs, *operands),
        lambda: _switch(idx - split, right_funcs, *operands),
    )


def apply_activation(idx, z, act_funcs):
    idx = _coerce_index(idx, device=z.device)
    if len(act_funcs) == 1:
        identity_mask = idx == -1
        activated = act_funcs[0](z)
        return torch.where(identity_mask, z, activated)

    return torch.cond(
        idx == -1,
        lambda: z.clone(),
        lambda: _switch(idx, act_funcs, z),
    )


def apply_aggregation(idx, z, agg_funcs, mask):
    idx = _coerce_index(idx, device=z.device)
    has_inputs = torch.any(mask)
    if len(agg_funcs) == 1:
        aggregated = agg_funcs[0](z, mask)
        return torch.where(has_inputs, aggregated, torch.zeros((), dtype=z.dtype, device=z.device))

    return torch.cond(
        has_inputs,
        lambda: _switch(idx, agg_funcs, z, mask),
        lambda: torch.zeros((), dtype=z.dtype, device=z.device),
    )


def get_func_name(func):
    name = func.__name__
    if name.endswith("_"):
        name = name[:-1]
    return name