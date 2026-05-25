import torch

from .base import BaseDistance
from ...gene import BaseGene
from ...utils import extract_gene_attrs


class DefaultDistance(BaseDistance):
    def __init__(self, compatibility_disjoint: float = 1.0, compatibility_weight: float = 0.4):
        self.compatibility_disjoint = compatibility_disjoint
        self.compatibility_weight = compatibility_weight

    def __call__(self, state, genome, nodes1, conns1, nodes2, conns2):
        node_distance = self.gene_distance(state, genome.node_gene, nodes1, nodes2)
        conn_distance = self.gene_distance(state, genome.conn_gene, conns1, conns2)
        return node_distance + conn_distance

    def gene_distance(self, state, gene: BaseGene, genes1, genes2):
        valid1 = [row for row in genes1 if not torch.isnan(row[0])]
        valid2 = [row for row in genes2 if not torch.isnan(row[0])]
        max_cnt = max(len(valid1), len(valid2))
        if max_cnt == 0:
            return torch.tensor(0.0, dtype=genes1.dtype, device=genes1.device)

        fixed_len = len(gene.fixed_attrs)
        gene_map2 = {
            tuple(int(v.item()) for v in row[:fixed_len]): extract_gene_attrs(gene, row)
            for row in valid2
        }

        homologous_distance = torch.tensor(0.0, dtype=genes1.dtype, device=genes1.device)
        homologous_cnt = 0
        for row in valid1:
            key = tuple(int(v.item()) for v in row[:fixed_len])
            if key in gene_map2:
                homologous_cnt += 1
                homologous_distance = homologous_distance + gene.distance(
                    state,
                    extract_gene_attrs(gene, row),
                    gene_map2[key].to(device=genes1.device, dtype=genes1.dtype),
                )

        non_homologous_cnt = len(valid1) + len(valid2) - 2 * homologous_cnt
        value = (
            non_homologous_cnt * self.compatibility_disjoint
            + homologous_distance * self.compatibility_weight
        )
        return value / max_cnt