from tensorneat import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.common import ACT
from tensorneat.genome import DefaultGenome
from tensorneat.problem import TargetTrackingEnv


def main():
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=16,
            species_size=4,
            survival_threshold=0.25,
            genome=DefaultGenome(
                num_inputs=1,
                num_outputs=1,
                max_nodes=6,
                max_conns=8,
                output_transform=ACT.sigmoid,
            ),
        ),
        problem=TargetTrackingEnv(max_step=6, repeat_times=1),
        generation_limit=6,
        fitness_target=5.9,
        seed=7,
    )
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    print(f"best fitness: {pipeline.best_fitness:.6f}")
    pipeline.show(state, best)


if __name__ == "__main__":
    main()