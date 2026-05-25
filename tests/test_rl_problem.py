import unittest

import torch

from tensorneat import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.common import ACT, State
from tensorneat.genome import DefaultGenome
from tensorneat.problem import TargetTrackingEnv


class RLProblemTests(unittest.TestCase):
    def test_rl_env_setup_and_evaluate_with_normalization(self):
        problem = TargetTrackingEnv(
            max_step=4,
            repeat_times=2,
            obs_normalization=True,
            sample_policy=lambda seed, obs: torch.zeros_like(obs),
            sample_episodes=3,
        )
        state = problem.setup(State(randkey=torch.tensor(7, dtype=torch.int64)))

        self.assertIn("problem_obs_mean", state)
        self.assertIn("problem_obs_std", state)

        fitness = problem.evaluate(
            state,
            torch.tensor(11, dtype=torch.int64),
            lambda current_state, params, obs: obs,
            None,
        )

        self.assertTrue(torch.isfinite(fitness))
        problem.show(state, torch.tensor(11, dtype=torch.int64), lambda current_state, params, obs: obs, None)

    def test_pipeline_runs_with_rl_problem(self):
        pipeline = Pipeline(
            algorithm=NEAT(
                genome=DefaultGenome(
                    num_inputs=1,
                    num_outputs=1,
                    max_nodes=6,
                    max_conns=8,
                    output_transform=ACT.sigmoid,
                ),
                pop_size=8,
                species_size=4,
            ),
            problem=TargetTrackingEnv(max_step=4, repeat_times=1),
            seed=3,
            generation_limit=2,
        )
        state = pipeline.setup()
        state, _, fitnesses = pipeline.step(state)

        self.assertEqual(fitnesses.shape, (8,))
        self.assertTrue(torch.isfinite(fitnesses).all())

        state, best = pipeline.auto_run(state)
        self.assertIsNotNone(best)


if __name__ == "__main__":
    unittest.main()