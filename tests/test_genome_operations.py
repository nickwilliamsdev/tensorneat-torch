import unittest

import torch

from tensorneat.common import State
from tensorneat.genome import DefaultGenome
from tensorneat.genome.operations import DefaultMutation
from tensorneat.genome.utils import valid_cnt


class GenomeOperationTests(unittest.TestCase):
    def test_mutation_add_node_keeps_feedforward_genome_runnable(self):
        genome = DefaultGenome(
            num_inputs=2,
            num_outputs=1,
            max_nodes=6,
            max_conns=8,
            mutation=DefaultMutation(node_add=1.0, node_delete=0.0, conn_add=0.0, conn_delete=0.0),
        )
        state = genome.setup(State())
        nodes, conns = genome.initialize(state, torch.Generator().manual_seed(0))

        mutated_nodes, mutated_conns = genome.execute_mutation(
            state,
            torch.Generator().manual_seed(1),
            nodes,
            conns,
            new_node_key=10,
            new_conn_keys=[20, 21, 22],
        )

        self.assertEqual(valid_cnt(mutated_nodes), valid_cnt(nodes) + 1)
        self.assertEqual(valid_cnt(mutated_conns), valid_cnt(conns) + 1)

        transformed = genome.transform(state, mutated_nodes, mutated_conns)
        out = genome.forward(state, transformed, torch.tensor([0.25, -0.5]))

        self.assertEqual(out.shape, (1,))
        self.assertTrue(torch.isfinite(out).all())

    def test_crossover_preserves_primary_parent_fixed_gene_keys(self):
        genome = DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=6, max_conns=8)
        state = genome.setup(State())
        nodes1, conns1 = genome.initialize(state, torch.Generator().manual_seed(2))
        nodes2 = nodes1.clone()
        conns2 = conns1.clone()

        nodes2[0, len(genome.node_gene.fixed_attrs) :] = genome.node_gene.new_random_attrs(None, torch.Generator().manual_seed(3))
        conns2[0, len(genome.conn_gene.fixed_attrs) :] = genome.conn_gene.new_random_attrs(None, torch.Generator().manual_seed(4))

        child_nodes, child_conns = genome.execute_crossover(
            state,
            torch.Generator().manual_seed(5),
            nodes1,
            conns1,
            nodes2,
            conns2,
        )

        self.assertTrue(
            torch.allclose(
                child_nodes[:, : len(genome.node_gene.fixed_attrs)],
                nodes1[:, : len(genome.node_gene.fixed_attrs)],
                equal_nan=True,
            )
        )
        self.assertTrue(
            torch.allclose(
                child_conns[:, : len(genome.conn_gene.fixed_attrs)],
                conns1[:, : len(genome.conn_gene.fixed_attrs)],
                equal_nan=True,
            )
        )
        self.assertTrue(
            torch.isclose(child_conns[0, -1], conns1[0, -1]) or torch.isclose(child_conns[0, -1], conns2[0, -1])
        )


if __name__ == "__main__":
    unittest.main()