import torch

from .default import DefaultMutation
from tensorneat.common import I_INF, check_cycles
from ...utils import add_conn, unflatten_conns


class RecurrentMutation(DefaultMutation):
    def __init__(self, *, p_recur: float = 0.1, max_conn_tries: int = 20, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= p_recur <= 1.0:
            raise ValueError("p_recur must be in [0, 1]")
        if max_conn_tries < 1:
            raise ValueError("max_conn_tries must be >= 1")
        self.p_recur = float(p_recur)
        self.max_conn_tries = int(max_conn_tries)

    def _mutate_add_conn(self, state, genome, generator, nodes, conns, new_conn_key):
        del state
        remain_conn_space = int(torch.sum(torch.isnan(conns[:, 0])).item())
        if remain_conn_space < 1:
            return nodes, conns

        u_conns = unflatten_conns(nodes, conns)
        conns_exist = u_conns != I_INF

        for _ in range(self.max_conn_tries):
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
                continue
            exists = torch.any((conns[:, 0] == i_key) & (conns[:, 1] == o_key))
            if bool(exists):
                continue

            forms_cycle = bool(check_cycles(nodes, conns_exist, int(from_idx), int(to_idx)))
            force_recur = float(torch.rand((), generator=generator, device=conns.device)) < self.p_recur
            if force_recur and not forms_cycle:
                continue

            if "historical_marker" in genome.conn_gene.fixed_attrs:
                fix_attrs = torch.tensor([float(i_key), float(o_key), float(new_conn_key[2])], device=conns.device, dtype=conns.dtype)
            else:
                fix_attrs = torch.tensor([float(i_key), float(o_key)], device=conns.device, dtype=conns.dtype)
            new_conns = add_conn(conns, fix_attrs, genome.conn_gene.new_zero_attrs(None).to(device=conns.device, dtype=conns.dtype))
            return nodes, new_conns

        return nodes, conns