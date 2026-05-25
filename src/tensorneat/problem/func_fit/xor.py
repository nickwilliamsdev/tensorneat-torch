import torch

from .func_fit import FuncFit


class XOR(FuncFit):
    @property
    def inputs(self):
        return torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

    @property
    def targets(self):
        return torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    @property
    def input_shape(self):
        return 4, 2

    @property
    def output_shape(self):
        return 4, 1