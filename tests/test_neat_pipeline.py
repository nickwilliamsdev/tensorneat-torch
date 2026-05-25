import unittest

import torch

from tensorneat import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.common import ACT, State
from tensorneat.genome import DefaultGenome, RecurrentGenome
from tensorneat.problem import XOR, XOR3d


class NeatPipelineTests(unittest.TestCase):
    def test_neat_setup_ask_tell_cycle(self):
        algo = NEAT(
            genome=DefaultGenome(num_inputs=2, num_outputs=1, max_nodes=8, max_conns=10),
            pop_size=6,
            species_size=4,
        )
        state = algo.setup(State(randkey=torch.tensor(123, dtype=torch.int64)))
        pop_nodes, pop_conns = algo.ask(state)

        self.assertEqual(pop_nodes.shape[0], 6)
        self.assertEqual(pop_conns.shape[0], 6)

        fitness = torch.linspace(0.0, 1.0, 6)
        next_state = algo.tell(state, fitness)
        next_nodes, next_conns = algo.ask(next_state)

        self.assertEqual(next_nodes.shape, pop_nodes.shape)
        self.assertEqual(next_conns.shape, pop_conns.shape)

    def test_pipeline_xor_step_and_show(self):
        pipeline = Pipeline(
            algorithm=NEAT(
                genome=DefaultGenome(
                    num_inputs=2,
                    num_outputs=1,
                    max_nodes=8,
                    max_conns=10,
                    output_transform=ACT.sigmoid,
                ),
                pop_size=32,
                species_size=6,
            ),
            problem=XOR(),
            seed=5,
            generation_limit=6,
        )
        state = pipeline.setup()
        state, pop, fitnesses = pipeline.step(state)
        first_best = float(torch.max(fitnesses))

        self.assertEqual(fitnesses.shape, (32,))

        state, best = pipeline.auto_run(state)
        self.assertIsNotNone(best)
        self.assertGreater(pipeline.best_fitness, first_best + 1e-4)
        pipeline.show(state, best)

    def test_recurrent_xor3d_short_run_improves_best_fitness(self):
        pipeline = Pipeline(
            algorithm=NEAT(
                genome=RecurrentGenome(
                    num_inputs=3,
                    num_outputs=1,
                    max_nodes=10,
                    max_conns=14,
                    output_transform=ACT.sigmoid,
                    activate_time=5,
                ),
                pop_size=32,
                species_size=6,
            ),
            problem=XOR3d(),
            seed=5,
            generation_limit=6,
        )
        state = pipeline.setup()
        state, _, fitnesses = pipeline.step(state)
        first_best = float(torch.max(fitnesses))

        state = pipeline.setup()
        state, best = pipeline.auto_run(state)

        self.assertIsNotNone(best)
        self.assertGreater(pipeline.best_fitness, first_best + 1e-4)


if __name__ == "__main__":
    unittest.main()