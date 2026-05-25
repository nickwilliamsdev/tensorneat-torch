from tensorneat import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.common import ACT
from tensorneat.genome import RecurrentGenome
from tensorneat.problem import XOR3d


def main():
    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=32,
            species_size=6,
            survival_threshold=0.25,
            genome=RecurrentGenome(
                num_inputs=3,
                num_outputs=1,
                max_nodes=10,
                max_conns=14,
                output_transform=ACT.sigmoid,
                activate_time=5,
            ),
        ),
        problem=XOR3d(),
        generation_limit=8,
        fitness_target=-1e-3,
        seed=5,
    )
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    print(f"best fitness: {pipeline.best_fitness:.6f}")
    pipeline.show(state, best)


if __name__ == "__main__":
    main()