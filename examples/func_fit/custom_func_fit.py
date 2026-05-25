import torch

from tensorneat import Pipeline
from tensorneat.algorithm import NEAT
from tensorneat.common import ACT, AGG
from tensorneat.genome import BiasNode, DefaultGenome
from tensorneat.problem import CustomFuncFit


def pagie_polynomial(inputs):
    x, y = inputs
    result = 1 / (1 + torch.pow(x, -4)) + 1 / (1 + torch.pow(y, -4))
    return torch.tensor([result], dtype=torch.float32)


def square(x):
    return x**2


def main():
    try:
        ACT.square
    except AttributeError:
        ACT.add_func("square", square)

    custom_problem = CustomFuncFit(
        func=pagie_polynomial,
        low_bounds=[-1, -1],
        upper_bounds=[1, 1],
        method="sample",
        num_samples=100,
    )

    pipeline = Pipeline(
        algorithm=NEAT(
            pop_size=32,
            species_size=6,
            survival_threshold=0.25,
            genome=DefaultGenome(
                num_inputs=2,
                num_outputs=1,
                max_nodes=10,
                max_conns=14,
                node_gene=BiasNode(
                    activation_options=[ACT.identity, ACT.inv, ACT.square],
                    aggregation_options=[AGG.sum, AGG.product],
                ),
                output_transform=ACT.identity,
            ),
        ),
        problem=custom_problem,
        generation_limit=8,
        fitness_target=-1e-3,
        seed=11,
    )
    state = pipeline.setup()
    state, best = pipeline.auto_run(state)
    print(f"best fitness: {pipeline.best_fitness:.6f}")
    pipeline.show(state, best)


if __name__ == "__main__":
    main()