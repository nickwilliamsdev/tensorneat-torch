import torch

from .base import BaseGenome
from .gene import DefaultConn, DefaultNode
from .operations import DefaultCrossover, DefaultDistance, DefaultMutation
from .utils import extract_gene_attrs, unflatten_conns
from tensorneat.common import I_INF, attach_with_inf, find_useful_nodes, topological_sort, topological_sort_python


class DefaultGenome(BaseGenome):
    network_type = "feedforward"

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
    ):
        super().__init__(
            num_inputs,
            num_outputs,
            max_nodes,
            max_conns,
            node_gene or DefaultNode(),
            conn_gene or DefaultConn(),
            mutation or DefaultMutation(),
            crossover or DefaultCrossover(),
            distance or DefaultDistance(),
            output_transform,
            input_transform,
            init_hidden_layers,
        )

    def transform(self, state, nodes, conns):
        del state
        u_conns = unflatten_conns(nodes, conns)
        conn_exist = u_conns != I_INF
        seqs = topological_sort(nodes, conn_exist)
        return seqs, nodes, conns, u_conns

    def forward(self, state, transformed, inputs):
        if self.input_transform is not None:
            inputs = self.input_transform(inputs)

        cal_seqs, nodes, conns, u_conns = transformed
        values = torch.full((self.max_nodes,), torch.nan, device=nodes.device, dtype=nodes.dtype)
        input_idx_tensor = torch.as_tensor(self.input_idx, device=nodes.device, dtype=torch.long)
        output_idx_tensor = torch.as_tensor(self.output_idx, device=nodes.device, dtype=torch.long)
        input_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=nodes.device)
        output_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=nodes.device)
        input_mask[input_idx_tensor] = True
        output_mask[output_idx_tensor] = True
        values[input_idx_tensor] = inputs.to(device=nodes.device, dtype=nodes.dtype)

        nodes_attrs = torch.stack([extract_gene_attrs(self.node_gene, node) for node in nodes])
        conns_attrs = torch.stack([extract_gene_attrs(self.conn_gene, conn) for conn in conns])
        node_indices = torch.arange(self.max_nodes, device=nodes.device, dtype=torch.long)

        for seq_idx in range(self.max_nodes):
            node_pos = cal_seqs[seq_idx].to(dtype=torch.long)
            safe_node_pos = torch.where(
                node_pos == I_INF,
                torch.zeros_like(node_pos),
                node_pos,
            )

            conn_indices = u_conns[:, safe_node_pos]
            src_values = values
            valid_mask = (conn_indices != I_INF) & (~torch.isnan(src_values))

            hit_attrs = attach_with_inf(conns_attrs, conn_indices)
            edge_messages = self.conn_gene.forward(state, hit_attrs, src_values)
            ins = torch.where(valid_mask, edge_messages, torch.full_like(edge_messages, torch.nan))
            candidate_value = self.node_gene.forward(
                state,
                nodes_attrs[safe_node_pos],
                ins,
                is_output_node=output_mask[safe_node_pos],
                valid_mask=valid_mask,
            )
            should_update = (node_pos != I_INF) & (~input_mask[safe_node_pos]) & torch.any(valid_mask)
            update_mask = node_indices == safe_node_pos
            replacement = torch.where(should_update, candidate_value, values[safe_node_pos])
            values = torch.where(update_mask, replacement.expand_as(values), values)

        outputs = values[output_idx_tensor]
        if self.output_transform is None:
            return outputs
        return self.output_transform(outputs)

    def network_dict(self, state, nodes, conns):
        network = super().network_dict(state, nodes, conns)
        topo_order, topo_layers = topological_sort_python(set(network["nodes"]), set(network["conns"]))
        network["topo_order"] = topo_order
        network["topo_layers"] = topo_layers
        network["useful_nodes"] = find_useful_nodes(
            set(network["nodes"]),
            set(network["conns"]),
            set(self.output_idx),
        )
        return network

    def sympy_func(self, *args, **kwargs):
        raise NotImplementedError("Sympy export for DefaultGenome is not ported yet")

    def visualize(self, *args, **kwargs):
        raise NotImplementedError("Visualization for DefaultGenome is not ported yet")