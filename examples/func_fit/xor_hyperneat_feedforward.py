from tensorneat import Pipeline
from tensorneat.algorithm import HyperNEATFeedForward, MLPSubstrate, NEAT
from tensorneat.common import ACT
from tensorneat.genome import DefaultGenome
from tensorneat.problem import XOR


def main():
    pipeline = Pipeline(
        algorithm=HyperNEATFeedForward(
            substrate=MLPSubstrate(layers=[3, 4, 1], coor_range=(-3.0, 3.0, -3.0, 3.0)),
            neat=NEAT(
                pop_size=32,
                species_size=6,
                survival_threshold=0.25,
                genome=DefaultGenome(
                    num_inputs=4,
                    num_outputs=1,
                    max_nodes=10,
                    max_conns=16,
                    output_transform=ACT.tanh,
                ),
            ),
            activation=ACT.tanh,
            output_transform=ACT.sigmoid,
        ),
        problem=XOR(),
        generation_limit=8,
        fitness_target=-1e-3,
        seed=7,
    )
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    print(f"best fitness: {pipeline.best_fitness:.6f}")
    pipeline.show(state, best)


if __name__ == "__main__":
    main()