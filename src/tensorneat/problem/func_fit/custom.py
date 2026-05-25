from typing import Callable, List, Tuple, Union

import numpy as np
import torch

from .func_fit import FuncFit


class CustomFuncFit(FuncFit):
    def __init__(
        self,
        func: Callable,
        low_bounds: Union[List, Tuple, torch.Tensor],
        upper_bounds: Union[List, Tuple, torch.Tensor],
        method: str = "sample",
        num_samples: int = 100,
        step_size: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        if isinstance(low_bounds, (list, tuple)):
            low_bounds = torch.tensor(low_bounds, dtype=torch.float32)
        if isinstance(upper_bounds, (list, tuple)):
            upper_bounds = torch.tensor(upper_bounds, dtype=torch.float32)

        try:
            func(low_bounds)
        except Exception as exc:
            raise ValueError(f"func(low_bounds) raised an exception: {exc}") from exc

        if low_bounds.shape != upper_bounds.shape:
            raise ValueError("low_bounds and upper_bounds must have the same shape")
        if method not in {"sample", "grid"}:
            raise ValueError("method must be 'sample' or 'grid'")

        self.func = func
        self.low_bounds = low_bounds.to(dtype=torch.float32)
        self.upper_bounds = upper_bounds.to(dtype=torch.float32)
        self.method = method
        self.num_samples = num_samples
        self.step_size = step_size.to(dtype=torch.float32) if step_size is not None else None
        self.generate_dataset()
        super().__init__(*args, **kwargs)

    def generate_dataset(self):
        if self.method == "sample":
            if self.num_samples <= 0:
                raise ValueError("num_samples must be positive")
            inputs = torch.zeros((self.num_samples, self.low_bounds.shape[0]), dtype=torch.float32)
            for idx in range(self.low_bounds.shape[0]):
                inputs[:, idx] = torch.empty(self.num_samples).uniform_(float(self.low_bounds[idx]), float(self.upper_bounds[idx]))
        elif self.method == "grid":
            if self.step_size is None:
                raise ValueError("step_size must be provided when method is 'grid'")
            if self.step_size.shape != self.low_bounds.shape:
                raise ValueError("step_size must have the same shape as low_bounds")
            if not bool(torch.all(self.step_size > 0)):
                raise ValueError("step_size must be positive")

            inputs = torch.zeros((1, 1), dtype=torch.float32)
            for idx in range(self.low_bounds.shape[0]):
                new_col = torch.arange(float(self.low_bounds[idx]), float(self.upper_bounds[idx]), float(self.step_size[idx]), dtype=torch.float32)
                inputs = cartesian_product(inputs, new_col[:, None])
            inputs = inputs[:, 1:]
        else:
            raise ValueError(f"Unknown method: {self.method}")

        outputs = torch.stack([torch.as_tensor(self.func(sample), dtype=torch.float32) for sample in inputs])
        self.data_inputs = inputs
        self.data_outputs = outputs

    @property
    def inputs(self):
        return self.data_inputs

    @property
    def targets(self):
        return self.data_outputs

    @property
    def input_shape(self):
        return tuple(self.data_inputs.shape)

    @property
    def output_shape(self):
        return tuple(self.data_outputs.shape)


def cartesian_product(arr1, arr2):
    if arr1.ndim != arr2.ndim:
        raise ValueError("arr1 and arr2 must have the same number of dimensions")
    if arr1.ndim > 2:
        raise ValueError("arr1 and arr2 must have at most 2 dimensions")

    len1 = arr1.shape[0]
    len2 = arr2.shape[0]
    repeated_arr1 = arr1.repeat_interleave(len2, dim=0)
    tiled_arr2 = arr2.repeat((len1, 1))
    return torch.cat((repeated_arr1, tiled_arr2), dim=1)