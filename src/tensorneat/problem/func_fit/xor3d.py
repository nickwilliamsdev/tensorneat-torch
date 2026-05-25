import torch

from .func_fit import FuncFit


class XOR3d(FuncFit):
    @property
    def inputs(self):
        return torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.float32,
        )

    @property
    def targets(self):
        return torch.tensor([[0], [1], [1], [0], [1], [0], [0], [1]], dtype=torch.float32)

    @property
    def input_shape(self):
        return 8, 3

    @property
    def output_shape(self):
        return 8, 1