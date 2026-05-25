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


def _coerce_index(idx):
    if isinstance(idx, torch.Tensor):
        if idx.ndim != 0:
            raise ValueError(f"Expected a scalar index tensor, got shape {tuple(idx.shape)}")
        return idx.to(dtype=torch.long)
    return torch.tensor(idx, dtype=torch.long)


def apply_activation(idx, z, act_funcs):
    idx = _coerce_index(idx)
    candidates = torch.stack([func(z) for func in act_funcs], dim=0)
    identity_mask = idx == -1
    safe_idx = torch.where(identity_mask, torch.zeros_like(idx), idx)
    selected = candidates[safe_idx]
    return torch.where(identity_mask, z, selected)


def apply_aggregation(idx, z, agg_funcs, mask):
    idx = _coerce_index(idx)
    candidates = torch.stack([func(z, mask) for func in agg_funcs], dim=0)
    has_inputs = torch.any(mask)
    selected = candidates[idx]
    return torch.where(has_inputs, selected, torch.zeros((), dtype=z.dtype, device=z.device))


def get_func_name(func):
    name = func.__name__
    if name.endswith("_"):
        name = name[:-1]
    return name