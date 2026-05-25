import torch

from .base import BaseCrossover
from ...gene import BaseGene
from ...utils import extract_gene_attrs, set_gene_attrs


class DefaultCrossover(BaseCrossover):
    def __call__(self, state, genome, generator, nodes1, conns1, nodes2, conns2):
        new_node_attrs = torch.stack(
            [
                create_new_gene(
                    state,
                    generator,
                    genome.node_gene,
                    nodes1[idx, : len(genome.node_gene.fixed_attrs)],
                    extract_gene_attrs(genome.node_gene, nodes1[idx]),
                    nodes2[:, : len(genome.node_gene.fixed_attrs)],
                    torch.stack([extract_gene_attrs(genome.node_gene, row) for row in nodes2]),
                )
                for idx in range(nodes1.shape[0])
            ]
        )
        new_nodes = torch.stack(
            [set_gene_attrs(genome.node_gene, nodes1[idx], new_node_attrs[idx]) for idx in range(nodes1.shape[0])]
        )

        new_conn_attrs = torch.stack(
            [
                create_new_gene(
                    state,
                    generator,
                    genome.conn_gene,
                    conns1[idx, : len(genome.conn_gene.fixed_attrs)],
                    extract_gene_attrs(genome.conn_gene, conns1[idx]),
                    conns2[:, : len(genome.conn_gene.fixed_attrs)],
                    torch.stack([extract_gene_attrs(genome.conn_gene, row) for row in conns2]),
                )
                for idx in range(conns1.shape[0])
            ]
        )
        new_conns = torch.stack(
            [set_gene_attrs(genome.conn_gene, conns1[idx], new_conn_attrs[idx]) for idx in range(conns1.shape[0])]
        )

        return new_nodes, new_conns


def create_new_gene(state, generator, gene: BaseGene, gene_key, gene_attrs, genes_keys, genes_attrs):
    homologous_idx = None
    if not torch.isnan(gene_key[0]):
        for idx in range(genes_keys.shape[0]):
            if torch.isnan(genes_keys[idx, 0]):
                continue
            if bool(torch.all(gene_key == genes_keys[idx])):
                homologous_idx = idx
                break

    if homologous_idx is None:
        return gene_attrs
    return gene.crossover(state, generator, gene_attrs, genes_attrs[homologous_idx])