from tensorneat import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.common import ACT
from tensorneat.genome import DefaultGenome
from tensorneat.problem import XOR


def main():
    algorithm = NEAT(
        pop_size=64,
        species_size=8,
        survival_threshold=0.2,
        genome=DefaultGenome(
            num_inputs=2,
            num_outputs=1,
            max_nodes=8,
            max_conns=12,
            output_transform=ACT.sigmoid,
        ),
    )
    pipeline = Pipeline(
        algorithm=algorithm,
        problem=XOR(),
        generation_limit=10,
        fitness_target=-1e-3,
        seed=42,
    )
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    print(f"best fitness: {pipeline.best_fitness:.6f}")
    pipeline.show(state, best)


if __name__ == "__main__":
    main()