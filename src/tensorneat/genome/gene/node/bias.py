from typing import Callable, Optional, Sequence, Union

import sympy as sp
import torch

from tensorneat.common import (
    ACT,
    AGG,
    apply_activation,
    apply_aggregation,
    get_func_name,
    mutate_float,
    mutate_int,
)

from .base import BaseNode


class BiasNode(BaseNode):
    """
    Default node gene, with the same behavior as in NEAT-python.
    The attribute response is removed.
    """

    custom_attrs = ["bias", "aggregation", "activation"]

    def __init__(
        self,
        bias_init_mean: float = 0.0,
        bias_init_std: float = 1.0,
        bias_mutate_power: float = 0.15,
        bias_mutate_rate: float = 0.2,
        bias_replace_rate: float = 0.015,
        bias_lower_bound: float = -5,
        bias_upper_bound: float = 5,
        aggregation_default: Optional[Callable] = None,
        aggregation_options: Union[Callable, Sequence[Callable]] = AGG.sum,
        aggregation_replace_rate: float = 0.1,
        activation_default: Optional[Callable] = None,
        activation_options: Union[Callable, Sequence[Callable]] = ACT.sigmoid,
        activation_replace_rate: float = 0.1,
    ):
        super().__init__()

        if callable(aggregation_options):
            aggregation_options = [aggregation_options]
        if callable(activation_options):
            activation_options = [activation_options]

        if aggregation_default is None:
            aggregation_default = aggregation_options[0]
        if activation_default is None:
            activation_default = activation_options[0]

        self.bias_init_mean = bias_init_mean
        self.bias_init_std = bias_init_std
        self.bias_mutate_power = bias_mutate_power
        self.bias_mutate_rate = bias_mutate_rate
        self.bias_replace_rate = bias_replace_rate
        self.bias_lower_bound = bias_lower_bound
        self.bias_upper_bound = bias_upper_bound

        self.aggregation_default = aggregation_options.index(aggregation_default)
        self.aggregation_options = list(aggregation_options)
        self.aggregation_indices = torch.arange(len(aggregation_options), dtype=torch.long)
        self.aggregation_replace_rate = aggregation_replace_rate

        self.activation_default = activation_options.index(activation_default)
        self.activation_options = list(activation_options)
        self.activation_indices = torch.arange(len(activation_options), dtype=torch.long)
        self.activation_replace_rate = activation_replace_rate

    def new_identity_attrs(self, state):
        del state
        return torch.tensor([0.0, float(self.aggregation_default), -1.0], dtype=torch.float32)

    def new_random_attrs(self, state, generator):
        del state
        bias = torch.randn((), generator=generator) * self.bias_init_std + self.bias_init_mean
        bias = torch.clamp(bias, self.bias_lower_bound, self.bias_upper_bound)
        aggregation = torch.randint(
            low=0,
            high=len(self.aggregation_options),
            size=(),
            generator=generator,
        )
        activation = torch.randint(
            low=0,
            high=len(self.activation_options),
            size=(),
            generator=generator,
        )
        return torch.stack(
            [
                bias.to(dtype=torch.float32),
                aggregation.to(dtype=torch.float32),
                activation.to(dtype=torch.float32),
            ]
        )

    def mutate(self, state, generator, attrs):
        del state
        bias, aggregation, activation = attrs

        bias = mutate_float(
            generator,
            bias,
            self.bias_init_mean,
            self.bias_init_std,
            self.bias_mutate_power,
            self.bias_mutate_rate,
            self.bias_replace_rate,
        )
        bias = torch.clamp(bias, self.bias_lower_bound, self.bias_upper_bound)
        aggregation = mutate_int(
            generator,
            aggregation,
            self.aggregation_indices.to(device=attrs.device),
            self.aggregation_replace_rate,
        )
        activation = mutate_int(
            generator,
            activation,
            self.activation_indices.to(device=attrs.device),
            self.activation_replace_rate,
        )

        return torch.stack(
            [
                bias.to(dtype=attrs.dtype),
                aggregation.to(dtype=attrs.dtype),
                activation.to(dtype=attrs.dtype),
            ]
        )

    def distance(self, state, attrs1, attrs2):
        del state
        bias1, aggregation1, activation1 = attrs1
        bias2, aggregation2, activation2 = attrs2
        return (
            torch.abs(bias1 - bias2)
            + (aggregation1 != aggregation2).to(dtype=attrs1.dtype)
            + (activation1 != activation2).to(dtype=attrs1.dtype)
        )

    def forward(self, state, attrs, inputs, is_output_node=False, valid_mask=None):
        del state
        bias, aggregation, activation = attrs
        if valid_mask is None:
            valid_mask = ~torch.isnan(inputs)
        z = apply_aggregation(aggregation, inputs, self.aggregation_options, valid_mask)
        z = bias + z
        activated = apply_activation(activation, z, self.activation_options)
        output_mask = torch.as_tensor(is_output_node, device=z.device, dtype=torch.bool)
        return torch.where(output_mask, z, activated)

    def repr(self, state, node, precision=2, idx_width=3, func_width=8):
        del state
        idx, bias, aggregation, activation = node
        act_func = ACT.identity if int(activation) == -1 else self.activation_options[int(activation)]
        return "{}(idx={:<{idx_width}}, bias={:<{float_width}}, aggregation={:<{func_width}}, activation={:<{func_width}})".format(
            self.__class__.__name__,
            int(idx),
            round(float(bias), precision),
            get_func_name(self.aggregation_options[int(aggregation)]),
            get_func_name(act_func),
            idx_width=idx_width,
            float_width=precision + 3,
            func_width=func_width,
        )

    def to_dict(self, state, node):
        del state
        idx, bias, aggregation, activation = node
        act_func = ACT.identity if int(activation) == -1 else self.activation_options[int(activation)]
        return {
            "idx": int(idx),
            "bias": torch.tensor(float(bias), dtype=torch.float32),
            "agg": get_func_name(self.aggregation_options[int(aggregation)]),
            "act": get_func_name(act_func),
        }

    def sympy_func(self, state, node_dict, inputs, is_output_node=False):
        del state
        bias = sp.symbols(f"n_{node_dict['idx']}_b")
        z = AGG.obtain_sympy(node_dict["agg"])(inputs)
        z = bias + z
        if not is_output_node:
            z = ACT.obtain_sympy(node_dict["act"])(z)
        return z, {bias: node_dict["bias"]}