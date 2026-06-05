import sympy as sp
import torch

from tensorneat.common import mutate_float

from .base import BaseConn


class DefaultConn(BaseConn):
    "Default connection gene, with the same behavior as in NEAT-python."

    custom_attrs = ["weight"]

    def __init__(
        self,
        weight_init_mean: float = 0.0,
        weight_init_std: float = 1.0,
        weight_mutate_power: float = 0.15,
        weight_mutate_rate: float = 0.2,
        weight_replace_rate: float = 0.015,
        weight_lower_bound: float = -5.0,
        weight_upper_bound: float = 5.0,
    ):
        super().__init__()
        self.weight_init_mean = weight_init_mean
        self.weight_init_std = weight_init_std
        self.weight_mutate_power = weight_mutate_power
        self.weight_mutate_rate = weight_mutate_rate
        self.weight_replace_rate = weight_replace_rate
        self.weight_lower_bound = weight_lower_bound
        self.weight_upper_bound = weight_upper_bound

    def new_zero_attrs(self, state):
        del state
        return torch.tensor([0.0], dtype=torch.float32)

    def new_identity_attrs(self, state):
        del state
        return torch.tensor([1.0], dtype=torch.float32)

    def new_random_attrs(self, state, generator):
        del state
        weight = torch.randn((), generator=generator, device=generator.device) * self.weight_init_std + self.weight_init_mean
        weight = torch.clamp(weight, self.weight_lower_bound, self.weight_upper_bound)
        return torch.tensor([float(weight)], dtype=torch.float32)

    def mutate(self, state, generator, attrs):
        del state
        weight = mutate_float(
            generator,
            attrs[0],
            self.weight_init_mean,
            self.weight_init_std,
            self.weight_mutate_power,
            self.weight_mutate_rate,
            self.weight_replace_rate,
        )
        weight = torch.clamp(weight, self.weight_lower_bound, self.weight_upper_bound)
        return torch.tensor([float(weight)], dtype=attrs.dtype, device=attrs.device)

    def distance(self, state, attrs1, attrs2):
        del state
        return torch.abs(attrs1[0] - attrs2[0])

    def forward(self, state, attrs, inputs):
        del state
        weight = attrs[..., 0]
        safe_w = torch.where(torch.isnan(weight), torch.zeros_like(weight), weight)
        safe_i = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
        return safe_i * safe_w

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        del state, func_width
        in_idx, out_idx, weight = conn
        return "{}(in: {:<{idx_width}}, out: {:<{idx_width}}, weight: {:<{float_width}})".format(
            self.__class__.__name__,
            int(in_idx),
            int(out_idx),
            round(float(weight), precision),
            idx_width=idx_width,
            float_width=precision + 3,
        )

    def to_dict(self, state, conn):
        del state
        return {
            "in": int(conn[0]),
            "out": int(conn[1]),
            "weight": torch.tensor(float(conn[2]), dtype=torch.float32),
        }

    def sympy_func(self, state, conn_dict, inputs, precision=None):
        del state, precision
        weight = sp.symbols(f"c_{conn_dict['in']}_{conn_dict['out']}_w")
        return inputs * weight, {weight: conn_dict["weight"]}