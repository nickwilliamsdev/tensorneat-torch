import torch

from .base import BaseSubstrate


class DefaultSubstrate(BaseSubstrate):
    connection_type = "recurrent"

    def __init__(self, num_inputs, num_outputs, coors, nodes, conns):
        self.inputs = num_inputs
        self.outputs = num_outputs
        self.coors = torch.as_tensor(coors, dtype=torch.float32)
        self.nodes = torch.as_tensor(nodes, dtype=torch.float32)
        self.conns = torch.as_tensor(conns, dtype=torch.float32)

    def make_nodes(self, query_res):
        return self.nodes.to(device=query_res.device, dtype=query_res.dtype).clone()

    def make_conns(self, query_res):
        conns = self.conns.to(device=query_res.device, dtype=query_res.dtype).clone()
        conns[:, -1] = query_res.reshape(-1)
        return conns

    @property
    def query_coors(self):
        return self.coors

    @property
    def num_inputs(self):
        return self.inputs

    @property
    def num_outputs(self):
        return self.outputs

    @property
    def nodes_cnt(self):
        return self.nodes.shape[0]

    @property
    def conns_cnt(self):
        return self.conns.shape[0]