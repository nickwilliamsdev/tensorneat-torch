import argparse
import time

import torch

from tensorneat import Pipeline
from tensorneat.algorithm import HyperNEATFeedForward, MLPSubstrate, NEAT
from tensorneat.common import ACT
from tensorneat.genome import DefaultGenome
from tensorneat.problem import XOR


def run_once(device: str, pop_size: int, generations: int, seed: int) -> float:
    start = time.perf_counter()
    pipeline = Pipeline(
        algorithm=HyperNEATFeedForward(
            substrate=MLPSubstrate(layers=[3, 4, 1], coor_range=(-3.0, 3.0, -3.0, 3.0)),
            neat=NEAT(
                pop_size=pop_size,
                species_size=6,
                survival_threshold=0.25,
                genome=DefaultGenome(
                    num_inputs=4,
                    num_outputs=1,
                    max_nodes=10,
                    max_conns=16,
                    output_transform=ACT.tanh,
                ),
                device=device,
            ),
            activation=ACT.tanh,
            output_transform=ACT.sigmoid,
        ),
        problem=XOR(),
        generation_limit=generations,
        fitness_target=-1e-3,
        seed=seed,
    )
    state = pipeline.setup()
    pipeline.auto_run(state)
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark HyperNEAT XOR on CPU and CUDA")
    parser.add_argument("--pop-size", type=int, default=128)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        times = []
        for rep in range(args.repeats):
            elapsed = run_once(device, args.pop_size, args.generations, args.seed + rep)
            times.append(elapsed)
        mean = sum(times) / len(times)
        print(
            f"device={device} pop_size={args.pop_size} generations={args.generations} repeats={args.repeats} "
            f"mean_s={mean:.3f} times={[round(v, 3) for v in times]}"
        )


if __name__ == "__main__":
    main()
