import torch

from .base import BaseGenome
from .gene import DefaultConn, DefaultNode
from .operations import DefaultCrossover, DefaultDistance, RecurrentMutation
from .utils import extract_gene_attrs, unflatten_conns
from tensorneat.common import I_INF, attach_with_inf


class RecurrentGenome(BaseGenome):
    network_type = "recurrent"

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        max_nodes=50,
        max_conns=100,
        node_gene=None,
        conn_gene=None,
        mutation=None,
        crossover=None,
        distance=None,
        output_transform=None,
        input_transform=None,
        init_hidden_layers=(),
        activate_time=10,
    ):
        super().__init__(
            num_inputs,
            num_outputs,
            max_nodes,
            max_conns,
            node_gene or DefaultNode(),
            conn_gene or DefaultConn(),
            mutation or RecurrentMutation(),
            crossover or DefaultCrossover(),
            distance or DefaultDistance(),
            output_transform,
            input_transform,
            init_hidden_layers,
        )
        self.activate_time = activate_time

    def transform(self, state, nodes, conns):
        del state
        u_conns = unflatten_conns(nodes, conns)
        return nodes, conns, u_conns

    def forward(self, state, transformed, inputs):
        nodes, conns, u_conns = transformed
        values = torch.full((self.max_nodes,), torch.nan, device=nodes.device, dtype=nodes.dtype)
        nodes_attrs = torch.stack([extract_gene_attrs(self.node_gene, node) for node in nodes])
        conns_attrs = torch.stack([extract_gene_attrs(self.conn_gene, conn) for conn in conns])
        node_exists = ~torch.isnan(nodes[:, 0])
        input_idx_tensor = torch.as_tensor(self.input_idx, device=nodes.device, dtype=torch.long)
        output_idx_tensor = torch.as_tensor(self.output_idx, device=nodes.device, dtype=torch.long)
        output_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=nodes.device)
        output_mask[output_idx_tensor] = True
        node_indices = torch.arange(self.max_nodes, device=nodes.device, dtype=torch.long)

        for _ in range(self.activate_time):
            if self.input_transform is not None:
                current_inputs = self.input_transform(inputs)
            else:
                current_inputs = inputs
            source_values = values.clone()
            source_values[input_idx_tensor] = current_inputs.to(device=nodes.device, dtype=nodes.dtype)
            new_values = source_values.clone()

            for node_pos in range(self.max_nodes):
                conn_indices = u_conns[:, node_pos]
                src_values = source_values
                valid_mask = (conn_indices != I_INF) & (~torch.isnan(src_values))
                hit_attrs = attach_with_inf(conns_attrs, conn_indices)
                edge_messages = self.conn_gene.forward(state, hit_attrs, src_values)
                ins = torch.where(valid_mask, edge_messages, torch.full_like(edge_messages, torch.nan))
                candidate_value = self.node_gene.forward(
                    state,
                    nodes_attrs[node_pos],
                    ins,
                    is_output_node=output_mask[node_pos],
                    valid_mask=valid_mask,
                )
                should_update = node_exists[node_pos] & torch.any(valid_mask)
                update_mask = node_indices == node_pos
                replacement = torch.where(should_update, candidate_value, new_values[node_pos])
                new_values = torch.where(update_mask, replacement.expand_as(new_values), new_values)

            values = new_values

        outputs = values[output_idx_tensor]
        if self.output_transform is None:
            return outputs
        return self.output_transform(outputs)

    def sympy_func(self, *args, **kwargs):
        raise ValueError("Sympy function is not supported for RecurrentGenome yet")

    def visualize(self, *args, **kwargs):
        raise ValueError("Visualize function is not supported for RecurrentGenome yet")