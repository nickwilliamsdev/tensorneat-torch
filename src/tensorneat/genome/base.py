from typing import Callable, Sequence

import numpy as np
import torch

from .gene import BaseConn, BaseNode
from .operations import BaseCrossover, BaseDistance, BaseMutation
from .utils import re_cound_idx, valid_cnt
from tensorneat.common import State, StatefulBaseClass, hash_array


class BaseGenome(StatefulBaseClass):
    network_type = None

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes: int,
        max_conns: int,
        node_gene: BaseNode,
        conn_gene: BaseConn,
        mutation: BaseMutation | None = None,
        crossover: BaseCrossover | None = None,
        distance: BaseDistance | None = None,
        output_transform: Callable | None = None,
        input_transform: Callable | None = None,
        init_hidden_layers: Sequence[int] = (),
    ):
        if input_transform is not None:
            try:
                input_transform(torch.zeros(num_inputs))
            except Exception as exc:
                raise ValueError(f"Input transform function failed: {exc}") from exc

        if output_transform is not None:
            try:
                output_transform(torch.zeros(num_outputs))
            except Exception as exc:
                raise ValueError(f"Output transform function failed: {exc}") from exc

        all_layers = [num_inputs] + list(init_hidden_layers) + [num_outputs]
        layer_indices = []
        next_index = 0
        for layer_size in all_layers:
            layer_indices.append(list(range(next_index, next_index + layer_size)))
            next_index += layer_size

        all_init_nodes = []
        all_init_conns = []
        for layer_idx in range(len(layer_indices) - 1):
            in_layer = layer_indices[layer_idx]
            out_layer = layer_indices[layer_idx + 1]
            for in_idx in in_layer:
                for out_idx in out_layer:
                    all_init_conns.append((in_idx, out_idx))
            all_init_nodes.extend(in_layer)
        all_init_nodes.extend(layer_indices[-1])

        if max_nodes < len(all_init_nodes):
            raise ValueError(
                f"max_nodes={max_nodes} must be >= initial nodes={len(all_init_nodes)}"
            )
        if max_conns < len(all_init_conns):
            raise ValueError(
                f"max_conns={max_conns} must be >= initial conns={len(all_init_conns)}"
            )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_nodes = max_nodes
        self.max_conns = max_conns
        self.node_gene = node_gene
        self.conn_gene = conn_gene
        self.mutation = mutation
        self.crossover = crossover
        self.distance = distance
        self.output_transform = output_transform
        self.input_transform = input_transform

        self.input_idx = np.array(layer_indices[0])
        self.output_idx = np.array(layer_indices[-1])
        self.all_init_nodes = np.array(all_init_nodes)
        self.all_init_conns = np.array(all_init_conns)

    def setup(self, state=State()):
        state = self.node_gene.setup(state)
        state = self.conn_gene.setup(state)
        if self.mutation is not None:
            state = self.mutation.setup(state)
        if self.crossover is not None:
            state = self.crossover.setup(state)
        if self.distance is not None:
            state = self.distance.setup(state)
        return state

    def transform(self, state, nodes, conns):
        raise NotImplementedError

    def forward(self, state, transformed, inputs):
        raise NotImplementedError

    def sympy_func(self, *args, **kwargs):
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        raise NotImplementedError

    def grad(self, state, nodes, conns, inputs, loss_fn):
        raise NotImplementedError("Gradient support is not ported yet")

    def execute_mutation(self, state, generator, nodes, conns, new_node_key, new_conn_keys):
        if self.mutation is None:
            raise NotImplementedError("Mutation operator has not been ported yet")
        return self.mutation(state, self, generator, nodes, conns, new_node_key, new_conn_keys)

    def execute_crossover(self, state, generator, nodes1, conns1, nodes2, conns2):
        if self.crossover is None:
            raise NotImplementedError("Crossover operator has not been ported yet")
        return self.crossover(state, self, generator, nodes1, conns1, nodes2, conns2)

    def execute_distance(self, state, nodes1, conns1, nodes2, conns2):
        if self.distance is None:
            raise NotImplementedError("Distance operator has not been ported yet")
        return self.distance(state, self, nodes1, conns1, nodes2, conns2)

    def initialize(self, state, generator, device=None, dtype=torch.float32):
        del state
        device = device or torch.device("cpu")

        nodes = torch.full((self.max_nodes, self.node_gene.length), torch.nan, device=device, dtype=dtype)
        conns = torch.full((self.max_conns, self.conn_gene.length), torch.nan, device=device, dtype=dtype)

        node_attrs = [self.node_gene.new_random_attrs(None, generator).to(device=device, dtype=dtype) for _ in self.all_init_nodes]
        if node_attrs:
            nodes[: len(self.all_init_nodes), 0] = torch.as_tensor(self.all_init_nodes, device=device, dtype=dtype)
            nodes[: len(self.all_init_nodes), 1:] = torch.stack(node_attrs)

        conn_attrs = [self.conn_gene.new_random_attrs(None, generator).to(device=device, dtype=dtype) for _ in self.all_init_conns]
        if conn_attrs:
            conns[: len(self.all_init_conns), :2] = torch.as_tensor(self.all_init_conns, device=device, dtype=dtype)
            if "historical_marker" in self.conn_gene.fixed_attrs:
                conns[: len(self.all_init_conns), 2] = torch.arange(len(self.all_init_conns), device=device, dtype=dtype)
            conns[: len(self.all_init_conns), len(self.conn_gene.fixed_attrs) :] = torch.stack(conn_attrs)

        return nodes, conns

    def network_dict(self, state, nodes, conns, whether_re_cound_idx=True):
        if whether_re_cound_idx:
            nodes, conns = re_cound_idx(nodes, conns, self.get_input_idx(), self.get_output_idx())
        return {"nodes": self._get_node_dict(state, nodes), "conns": self._get_conn_dict(state, conns)}

    def get_input_idx(self):
        return self.input_idx.tolist()

    def get_output_idx(self):
        return self.output_idx.tolist()

    def hash(self, nodes, conns):
        node_hashes = torch.vmap(self.node_gene.hash)(nodes)
        conn_hashes = torch.vmap(self.conn_gene.hash)(conns)
        combined = torch.cat([node_hashes, conn_hashes])
        return hash_array(combined)

    def repr(self, state, nodes, conns, precision=2):
        nodes_cnt, conns_cnt = valid_cnt(nodes), valid_cnt(conns)
        text = f"{self.__class__.__name__}(nodes={nodes_cnt}, conns={conns_cnt}):\n"
        text += "\tNodes:\n"
        for node in nodes.detach().cpu():
            if torch.isnan(node[0]):
                break
            text += f"\t\t{self.node_gene.repr(state, node, precision=precision)}"
            node_idx = int(node[0])
            if node_idx in self.input_idx:
                text += " (input)"
            elif node_idx in self.output_idx:
                text += " (output)"
            text += "\n"

        text += "\tConns:\n"
        for conn in conns.detach().cpu():
            if torch.isnan(conn[0]):
                break
            text += f"\t\t{self.conn_gene.repr(state, conn, precision=precision)}\n"
        return text

    def _get_conn_dict(self, state, conns):
        del state
        result = {}
        for conn in conns.detach().cpu():
            if torch.isnan(conn[0]):
                continue
            conn_dict = self.conn_gene.to_dict(None, conn)
            result[(conn_dict["in"], conn_dict["out"])] = conn_dict
        return result

    def _get_node_dict(self, state, nodes):
        del state
        result = {}
        for node in nodes.detach().cpu():
            if torch.isnan(node[0]):
                continue
            node_dict = self.node_gene.to_dict(None, node)
            result[node_dict["idx"]] = node_dict
        return result