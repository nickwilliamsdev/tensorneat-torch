from tensorneat import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.common import ACT
from tensorneat.genome import DefaultGenome
from tensorneat.problem import GymnasiumEnv


def main():
    try:
        problem = GymnasiumEnv(env_name="CartPole-v1", max_step=200, repeat_times=1)
    except ImportError as exc:
        raise SystemExit(
            "This example requires gymnasium. Install it in the project venv with: pip install gymnasium[classic-control]"
        ) from exc

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=32,
            species_size=6,
            survival_threshold=0.25,
            genome=DefaultGenome(
                num_inputs=4,
                num_outputs=2,
                max_nodes=10,
                max_conns=16,
                output_transform=ACT.identity,
            ),
        ),
        problem=problem,
        generation_limit=5,
        fitness_target=200.0,
        seed=13,
    )
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    print(f"best fitness: {pipeline.best_fitness:.6f}")
    pipeline.show(state, best)


if __name__ == "__main__":
    main()