import torch

from .base import BaseMutation
from tensorneat.common import I_INF, check_cycles, fetch_random
from ...utils import (
    add_conn,
    add_node,
    delete_conn_by_pos,
    delete_node_by_pos,
    extract_gene_attrs,
    set_gene_attrs,
    unflatten_conns,
)


class DefaultMutation(BaseMutation):
    def __init__(
        self,
        conn_add: float = 0.2,
        conn_delete: float = 0.2,
        node_add: float = 0.1,
        node_delete: float = 0.1,
    ):
        self.conn_add = conn_add
        self.conn_delete = conn_delete
        self.node_add = node_add
        self.node_delete = node_delete

    def __call__(self, state, genome, generator, nodes, conns, new_node_key, new_conn_key):
        nodes, conns = self.mutate_structure(
            state, genome, generator, nodes, conns, new_node_key, new_conn_key
        )
        nodes, conns = self.mutate_values(state, genome, generator, nodes, conns)
        return nodes, conns

    def mutate_structure(self, state, genome, generator, nodes, conns, new_node_key, new_conn_key):
        nodes = nodes.clone()
        conns = conns.clone()

        if self.node_add > 0 and float(torch.rand((), generator=generator)) < self.node_add:
            nodes, conns = self._mutate_add_node(state, genome, generator, nodes, conns, new_node_key, new_conn_key)

        if self.node_delete > 0 and float(torch.rand((), generator=generator)) < self.node_delete:
            nodes, conns = self._mutate_delete_node(state, genome, generator, nodes, conns)

        if self.conn_add > 0 and float(torch.rand((), generator=generator)) < self.conn_add:
            nodes, conns = self._mutate_add_conn(state, genome, generator, nodes, conns, new_conn_key)

        if self.conn_delete > 0 and float(torch.rand((), generator=generator)) < self.conn_delete:
            nodes, conns = self._mutate_delete_conn(generator, nodes, conns)

        return nodes, conns

    def mutate_values(self, state, genome, generator, nodes, conns):
        new_nodes = nodes.clone()
        for idx in range(genome.max_nodes):
            if torch.isnan(nodes[idx, 0]):
                continue
            attrs = extract_gene_attrs(genome.node_gene, nodes[idx])
            new_attrs = genome.node_gene.mutate(state, generator, attrs)
            new_nodes[idx] = set_gene_attrs(genome.node_gene, nodes[idx], new_attrs)

        new_conns = conns.clone()
        for idx in range(genome.max_conns):
            if torch.isnan(conns[idx, 0]):
                continue
            attrs = extract_gene_attrs(genome.conn_gene, conns[idx])
            new_attrs = genome.conn_gene.mutate(state, generator, attrs)
            new_conns[idx] = set_gene_attrs(genome.conn_gene, conns[idx], new_attrs)

        return new_nodes, new_conns

    def choose_node_key(
        self,
        generator,
        nodes,
        input_idx,
        output_idx,
        allow_input_keys=False,
        allow_output_keys=False,
    ):
        node_keys = nodes[:, 0]
        mask = ~torch.isnan(node_keys)

        input_idx_tensor = torch.as_tensor(input_idx, device=nodes.device, dtype=node_keys.dtype)
        output_idx_tensor = torch.as_tensor(output_idx, device=nodes.device, dtype=node_keys.dtype)

        if not allow_input_keys:
            mask = mask & ~torch.isin(node_keys, input_idx_tensor)
        if not allow_output_keys:
            mask = mask & ~torch.isin(node_keys, output_idx_tensor)

        idx = fetch_random(generator, mask)
        if int(idx) == I_INF:
            return torch.tensor(torch.nan, device=nodes.device, dtype=nodes.dtype), idx
        return nodes[int(idx), 0], idx

    def choose_connection_key(self, generator, conns):
        idx = fetch_random(generator, ~torch.isnan(conns[:, 0]))
        if int(idx) == I_INF:
            nan = torch.tensor(torch.nan, device=conns.device, dtype=conns.dtype)
            return nan, nan, idx
        idx_int = int(idx)
        return conns[idx_int, 0], conns[idx_int, 1], idx

    def _mutate_add_node(self, state, genome, generator, nodes, conns, new_node_key, new_conn_key):
        remain_node_space = int(torch.sum(torch.isnan(nodes[:, 0])).item())
        remain_conn_space = int(torch.sum(torch.isnan(conns[:, 0])).item())
        i_key, o_key, idx = self.choose_connection_key(generator, conns)
        if int(idx) == I_INF or remain_node_space < 1 or remain_conn_space < 2:
            return nodes, conns

        idx_int = int(idx)
        original_attrs = extract_gene_attrs(genome.conn_gene, conns[idx_int])
        new_conns = delete_conn_by_pos(conns, idx_int)

        new_node_key_tensor = torch.tensor([float(new_node_key)], device=nodes.device, dtype=nodes.dtype)
        new_nodes = add_node(nodes, new_node_key_tensor, genome.node_gene.new_identity_attrs(state).to(device=nodes.device, dtype=nodes.dtype))

        if "historical_marker" in genome.conn_gene.fixed_attrs:
            fix_attrs1 = torch.tensor([float(i_key), float(new_node_key), float(new_conn_key[0])], device=conns.device, dtype=conns.dtype)
            fix_attrs2 = torch.tensor([float(new_node_key), float(o_key), float(new_conn_key[1])], device=conns.device, dtype=conns.dtype)
        else:
            fix_attrs1 = torch.tensor([float(i_key), float(new_node_key)], device=conns.device, dtype=conns.dtype)
            fix_attrs2 = torch.tensor([float(new_node_key), float(o_key)], device=conns.device, dtype=conns.dtype)

        new_conns = add_conn(new_conns, fix_attrs1, genome.conn_gene.new_identity_attrs(state).to(device=conns.device, dtype=conns.dtype))
        new_conns = add_conn(new_conns, fix_attrs2, original_attrs)
        return new_nodes, new_conns

    def _mutate_delete_node(self, state, genome, generator, nodes, conns):
        del state
        key, idx = self.choose_node_key(
            generator,
            nodes,
            genome.input_idx,
            genome.output_idx,
            allow_input_keys=False,
            allow_output_keys=False,
        )
        if int(idx) == I_INF:
            return nodes, conns

        idx_int = int(idx)
        new_nodes = delete_node_by_pos(nodes, idx_int)
        delete_mask = ((conns[:, 0] == key) | (conns[:, 1] == key)).unsqueeze(1)
        new_conns = torch.where(delete_mask, torch.full_like(conns, torch.nan), conns)
        return new_nodes, new_conns

    def _mutate_add_conn(self, state, genome, generator, nodes, conns, new_conn_key):
        del state
        remain_conn_space = int(torch.sum(torch.isnan(conns[:, 0])).item())
        if remain_conn_space < 1:
            return nodes, conns

        i_key, from_idx = self.choose_node_key(
            generator,
            nodes,
            genome.input_idx,
            genome.output_idx,
            allow_input_keys=True,
            allow_output_keys=True,
        )
        o_key, to_idx = self.choose_node_key(
            generator,
            nodes,
            genome.input_idx,
            genome.output_idx,
            allow_input_keys=False,
            allow_output_keys=True,
        )
        if int(from_idx) == I_INF or int(to_idx) == I_INF:
            return nodes, conns

        exists = torch.any((conns[:, 0] == i_key) & (conns[:, 1] == o_key))
        if bool(exists):
            return nodes, conns

        if genome.network_type == "feedforward":
            u_conns = unflatten_conns(nodes, conns)
            conns_exist = u_conns != I_INF
            if bool(check_cycles(nodes, conns_exist, int(from_idx), int(to_idx))):
                return nodes, conns

        if "historical_marker" in genome.conn_gene.fixed_attrs:
            fix_attrs = torch.tensor([float(i_key), float(o_key), float(new_conn_key[2])], device=conns.device, dtype=conns.dtype)
        else:
            fix_attrs = torch.tensor([float(i_key), float(o_key)], device=conns.device, dtype=conns.dtype)
        new_conns = add_conn(conns, fix_attrs, genome.conn_gene.new_zero_attrs(None).to(device=conns.device, dtype=conns.dtype))
        return nodes, new_conns

    def _mutate_delete_conn(self, generator, nodes, conns):
        _, _, idx = self.choose_connection_key(generator, conns)
        if int(idx) == I_INF:
            return nodes, conns
        return nodes, delete_conn_by_pos(conns, int(idx))