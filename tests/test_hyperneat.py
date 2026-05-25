import unittest

import torch

from tensorneat.algorithm import DefaultSubstrate, HyperNEAT, HyperNEATFeedForward, MLPSubstrate, NEAT
from tensorneat.common import State
from tensorneat.genome import DefaultGenome


class HyperNeatTests(unittest.TestCase):
    def test_hyperneat_feedforward_transform_and_forward(self):
        substrate = MLPSubstrate([3, 1])
        base_neat = NEAT(
            genome=DefaultGenome(num_inputs=4, num_outputs=1, max_nodes=10, max_conns=12),
            pop_size=4,
            species_size=3,
        )
        hyper = HyperNEATFeedForward(substrate=substrate, neat=base_neat)
        state = hyper.setup(State(randkey=torch.tensor(17, dtype=torch.int64)))
        pop_nodes, pop_conns = hyper.ask(state)
        transformed = hyper.transform(state, (pop_nodes[0], pop_conns[0]))
        out = hyper.forward(state, transformed, torch.tensor([0.2, -0.3]))

        self.assertEqual(out.shape, (1,))

    def test_hyperneat_tell_cycle(self):
        substrate = MLPSubstrate([3, 1])
        base_neat = NEAT(
            genome=DefaultGenome(num_inputs=4, num_outputs=1, max_nodes=10, max_conns=12),
            pop_size=4,
            species_size=3,
        )
        hyper = HyperNEATFeedForward(substrate=substrate, neat=base_neat)
        state = hyper.setup(State(randkey=torch.tensor(19, dtype=torch.int64)))
        state = hyper.tell(state, torch.tensor([0.0, 1.0, 0.5, 0.25]))
        pop_nodes, pop_conns = hyper.ask(state)

        self.assertEqual(pop_nodes.shape[0], 4)
        self.assertEqual(pop_conns.shape[0], 4)

    def test_recurrent_hyperneat_transform_and_forward(self):
        substrate = DefaultSubstrate(
            num_inputs=2,
            num_outputs=1,
            coors=[[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
            nodes=[[0.0], [1.0], [2.0]],
            conns=[[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
        )
        base_neat = NEAT(
            genome=DefaultGenome(num_inputs=4, num_outputs=1, max_nodes=8, max_conns=10),
            pop_size=4,
            species_size=3,
        )
        hyper = HyperNEAT(substrate=substrate, neat=base_neat, activate_time=4)
        state = hyper.setup(State(randkey=torch.tensor(23, dtype=torch.int64)))
        pop_nodes, pop_conns = hyper.ask(state)
        transformed = hyper.transform(state, (pop_nodes[0], pop_conns[0]))
        out = hyper.forward(state, transformed, torch.tensor([0.2]))

        self.assertEqual(out.shape, (1,))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()