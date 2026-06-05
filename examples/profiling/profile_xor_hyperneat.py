import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile

from tensorneat import Pipeline
from tensorneat.algorithm import HyperNEATFeedForward, MLPSubstrate, NEAT
from tensorneat.common import ACT
from tensorneat.genome import DefaultGenome
from tensorneat.problem import XOR


def build_pipeline(device: str, pop_size: int, generation_limit: int, seed: int) -> Pipeline:
    return Pipeline(
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
        generation_limit=generation_limit,
        fitness_target=-1e-3,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile HyperNEAT XOR on CPU/CUDA")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--pop-size", type=int, default=128)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--with-stack", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    activities = [ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    pipeline = build_pipeline(
        device=args.device,
        pop_size=args.pop_size,
        generation_limit=args.generations,
        seed=args.seed,
    )

    t0 = time.perf_counter()
    state = pipeline.setup()

    # Warm up one step to reduce one-time initialization noise in profiling output.
    if args.generations > 0:
        state, _, _ = pipeline.step(state)
        remaining_generations = args.generations - 1
    else:
        remaining_generations = 0

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
    ) as prof:
        for _ in range(remaining_generations):
            state, _, _ = pipeline.step(state)

    total_s = time.perf_counter() - t0

    if args.device == "cuda":
        torch.cuda.synchronize()

    sort_by = "self_cuda_time_total" if args.device == "cuda" else "self_cpu_time_total"

    print(
        f"profile complete device={args.device} pop_size={args.pop_size} generations={args.generations} total_s={total_s:.3f}"
    )
    print(prof.key_averages().table(sort_by=sort_by, row_limit=args.top_k))


if __name__ == "__main__":
    main()
