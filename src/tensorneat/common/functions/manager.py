from functools import partial
from typing import Callable, Union

import numpy as np
import sympy as sp
import torch


class FunctionManager:
    def __init__(self, name2torch, name2sympy):
        self.name2torch = dict(name2torch)
        self.name2sympy = dict(name2sympy)
        for name, func in self.name2torch.items():
            setattr(self, name, func)

    def get_all_funcs(self):
        return [getattr(self, name) for name in self.name2torch]

    def add_func(self, name, func):
        if not callable(func):
            raise ValueError("The provided function is not callable")
        if name in self.name2torch:
            raise ValueError(f"The provided name={name} is already in use")

        self.name2torch[name] = func
        setattr(self, name, func)

    def update_sympy(self, name, sympy_cls: sp.Function):
        self.name2sympy[name] = sympy_cls

    def obtain_sympy(self, func: Union[str, Callable]):
        if isinstance(func, str):
            return self._obtain_sympy_by_name(func)

        if callable(func):
            for name, registered in self.name2torch.items():
                if registered == func:
                    return self._obtain_sympy_by_name(name)
            raise ValueError(f"Func {func} is not registered.")

        raise ValueError(f"Func {func} must be a string or callable.")

    def _obtain_sympy_by_name(self, name):
        if name not in self.name2sympy:
            raise ValueError(f"Func {name} doesn't have a sympy representation.")
        return self.name2sympy[name]

    def sympy_module(self, backend):
        if backend == "torch":
            numerical_backend = torch
        elif backend == "numpy":
            numerical_backend = np
        else:
            raise ValueError("backend must be 'torch' or 'numpy'")

        module = {}
        for sympy_cls in self.name2sympy.values():
            if hasattr(sympy_cls, "numerical_eval"):
                module[sympy_cls.__name__] = partial(
                    sympy_cls.numerical_eval,
                    backend=numerical_backend,
                )
        return module