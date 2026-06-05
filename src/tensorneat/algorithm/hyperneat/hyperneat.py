from typing import Callable

import torch

from .substrate import BaseSubstrate
from ..base import BaseAlgorithm
from ..neat import NEAT
from tensorneat.common import ACT, AGG, State
from tensorneat.genome import BaseConn, BaseNode, RecurrentGenome


class HyperNEAT(BaseAlgorithm):
    def __init__(
        self,
        substrate: BaseSubstrate,
        neat: NEAT,
        weight_threshold: float = 0.3,
        max_weight: float = 5.0,
        aggregation: Callable = AGG.sum,
        activation: Callable = ACT.sigmoid,
        activate_time: int = 10,
        output_transform: Callable = ACT.sigmoid,
    ):
        if substrate.query_coors.shape[1] != neat.num_inputs:
            raise ValueError("Query coors of Substrate should be equal to NEAT input size")

        self.substrate = substrate
        self.neat = neat
        self.weight_threshold = weight_threshold
        self.max_weight = max_weight
        self.hyper_genome = RecurrentGenome(
            num_inputs=substrate.num_inputs,
            num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt,
            max_conns=substrate.conns_cnt,
            node_gene=HyperNEATNode(aggregation, activation),
            conn_gene=HyperNEATConn(),
            activate_time=activate_time,
            output_transform=output_transform,
        )
        self.pop_size = neat.pop_size
        self._query_vmap_available = True

    def setup(self, state=State()):
        state = self.neat.setup(state)
        state = self.substrate.setup(state)
        return self.hyper_genome.setup(state)

    def ask(self, state):
        return self.neat.ask(state)

    def tell(self, state, fitness):
        return self.neat.tell(state, fitness)

    def transform(self, state, individual):
        transformed = self.neat.transform(state, individual)
        ref_tensor = self._find_first_tensor(transformed)
        if ref_tensor is None:
            query_coors = self.substrate.query_coors
        else:
            query_coors = self.substrate.query_coors.to(device=ref_tensor.device, dtype=ref_tensor.dtype)

        if self._query_vmap_available:
            evaluator = lambda coor: self.neat.forward(state, transformed, coor)
            try:
                query_res = torch.vmap(evaluator)(query_coors)
            except RuntimeError:
                self._query_vmap_available = False
                query_res = torch.stack([self.neat.forward(state, transformed, coor) for coor in query_coors])
        else:
            query_res = torch.stack([self.neat.forward(state, transformed, coor) for coor in query_coors])

        query_res = torch.where(
            (-self.weight_threshold < query_res) & (query_res < self.weight_threshold),
            torch.zeros_like(query_res),
            query_res,
        )
        query_res = torch.where(query_res > 0, query_res - self.weight_threshold, query_res)
        query_res = torch.where(query_res < 0, query_res + self.weight_threshold, query_res)
        query_res = query_res / (1 - self.weight_threshold) * self.max_weight

        h_nodes = self.substrate.make_nodes(query_res)
        h_conns = self.substrate.make_conns(query_res)
        return self.hyper_genome.transform(state, h_nodes, h_conns)

    def forward(self, state, transformed, inputs):
        inputs_with_bias = torch.cat([inputs, torch.ones(1, dtype=inputs.dtype, device=inputs.device)])
        return self.hyper_genome.forward(state, transformed, inputs_with_bias)

    @property
    def num_inputs(self):
        return self.substrate.num_inputs - 1

    @property
    def num_outputs(self):
        return self.substrate.num_outputs

    def show_details(self, state, fitness):
        return self.neat.show_details(state, fitness)

    def _find_first_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, tuple):
            for item in value:
                found = self._find_first_tensor(item)
                if found is not None:
                    return found
            return None
        if isinstance(value, list):
            for item in value:
                found = self._find_first_tensor(item)
                if found is not None:
                    return found
            return None
        return None


class HyperNEATNode(BaseNode):
    custom_attrs = []

    def __init__(self, aggregation=AGG.sum, activation=ACT.sigmoid):
        super().__init__()
        self.aggregation = aggregation
        self.activation = activation

    def new_identity_attrs(self, state):
        del state
        return torch.empty((0,), dtype=torch.float32)

    def new_random_attrs(self, state, generator):
        del state, generator
        return torch.empty((0,), dtype=torch.float32)

    def mutate(self, state, generator, attrs):
        del state, generator
        return attrs

    def crossover(self, state, generator, attrs1, attrs2):
        del state, generator, attrs2
        return attrs1

    def distance(self, state, attrs1, attrs2):
        del state, attrs1, attrs2
        return torch.tensor(0.0, dtype=torch.float32)

    def forward(self, state, attrs, inputs, is_output_node=False, valid_mask=None):
        del state, attrs, valid_mask
        aggregated = self.aggregation(inputs, ~torch.isnan(inputs)) if callable(getattr(self.aggregation, '__call__', None)) else self.aggregation(inputs)
        activated = self.activation(aggregated)
        output_mask = torch.as_tensor(is_output_node, device=aggregated.device, dtype=torch.bool)
        return torch.where(output_mask, aggregated, activated)

    def sympy_func(self, state, node_dict, inputs, is_output_node=False):
        raise NotImplementedError("HyperNEAT sympy export is not ported yet")


class HyperNEATConn(BaseConn):
    custom_attrs = ["weight"]

    def new_zero_attrs(self, state):
        del state
        return torch.tensor([0.0], dtype=torch.float32)

    def new_identity_attrs(self, state):
        del state
        return torch.tensor([1.0], dtype=torch.float32)

    def new_random_attrs(self, state, generator):
        del state, generator
        return torch.tensor([0.0], dtype=torch.float32)

    def mutate(self, state, generator, attrs):
        del state, generator
        return attrs

    def distance(self, state, attrs1, attrs2):
        del state
        return torch.abs(attrs1[0] - attrs2[0])

    def forward(self, state, attrs, inputs):
        del state
        return inputs * attrs[..., 0]

    def sympy_func(self, state, conn_dict, inputs):
        raise NotImplementedError("HyperNEAT sympy export is not ported yet")