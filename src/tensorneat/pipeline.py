import datetime
import json
import os
import time
from collections.abc import Sequence

import numpy as np
import torch

from tensorneat.algorithm import BaseAlgorithm
from tensorneat.common import State, StatefulBaseClass
from tensorneat.problem import BaseProblem


class Pipeline(StatefulBaseClass):
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        problem: BaseProblem,
        seed: int = 42,
        fitness_target: float = 1,
        generation_limit: int = 1000,
        is_save: bool = False,
        save_dir=None,
        show_problem_details: bool = False,
        using_multidevice: bool = False,
        eval_batch_size: int = None,
    ):
        self.algorithm = algorithm
        self.problem = problem
        self.seed = seed
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.pop_size = self.algorithm.pop_size
        self.eval_batch_size = eval_batch_size
        if eval_batch_size is not None and self.pop_size % eval_batch_size != 0:
            raise ValueError(
                f"pop_size ({self.pop_size}) must be divisible by eval_batch_size ({eval_batch_size})"
            )

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if algorithm.num_inputs != self.problem.input_shape[-1]:
            raise ValueError(
                f"algorithm input shape is {algorithm.num_inputs} but problem input shape is {self.problem.input_shape}"
            )

        self.best_genome = None
        self.best_fitness = float("-inf")
        self.generation_timestamp = None
        self._last_pop_transformed = None
        self._batched_eval_available = True
        self._compiled_batched_evaluator = None
        self.is_save = is_save
        self.show_problem_details = show_problem_details
        self.using_multidevice = using_multidevice

        if using_multidevice:
            raise NotImplementedError("Multi-device pipeline execution is not ported yet")

        if is_save:
            if save_dir is None:
                now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                self.save_dir = f"./{self.__class__.__name__} {now}"
            else:
                self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)
            self.genome_dir = os.path.join(self.save_dir, "genomes")
            os.makedirs(self.genome_dir, exist_ok=True)

    def setup(self, state=State()):
        print("initializing")
        if "randkey" in state:
            seeded_state = state
        else:
            seeded_state = state.register(randkey=torch.tensor(self.seed, dtype=torch.int64))

        seeded_state = self.algorithm.setup(seeded_state)
        seeded_state = self.problem.setup(seeded_state)

        if self.is_save:
            with open(os.path.join(self.save_dir, "config.txt"), "w", encoding="ascii") as handle:
                handle.write(json.dumps(self.show_config(), indent=4))
            with open(os.path.join(self.save_dir, "log.txt"), "w", encoding="ascii") as handle:
                handle.write("Generation,Max,Min,Mean,Std,Cost Time\n")

        print("initializing finished")
        return seeded_state

    def step(self, state):
        pop = self.algorithm.ask(state)
        pop_transformed = [self.algorithm.transform(state, (pop[0][idx], pop[1][idx])) for idx in range(self.pop_size)]
        self._last_pop_transformed = pop_transformed

        if isinstance(state.randkey, torch.Tensor):
            seed = state.randkey.to(dtype=torch.int64)
        else:
            seed = torch.tensor(int(state.randkey), dtype=torch.int64)
        fitnesses = self._evaluate_population(state, pop_transformed, seed)
        fitnesses = torch.where(torch.isnan(fitnesses), torch.full_like(fitnesses, -torch.inf), fitnesses)

        next_state = self.algorithm.tell(state, fitnesses)
        return next_state.update(randkey=seed + self.pop_size + 1), pop, fitnesses

    def auto_run(self, state):
        for _ in range(self.generation_limit):
            self.generation_timestamp = time.time()
            state, previous_pop, fitnesses = self.step(state)
            self.analysis(state, previous_pop, fitnesses.detach().cpu().numpy())
            if float(torch.max(fitnesses)) >= self.fitness_target:
                print("Fitness limit reached!")
                break

        if int(state.generation) >= self.generation_limit:
            print("Generation limit reached!")

        if self.is_save and self.best_genome is not None:
            best_nodes, best_conns = self.best_genome
            with open(os.path.join(self.genome_dir, "best_genome.npz"), "wb") as handle:
                np.savez(
                    handle,
                    nodes=best_nodes.detach().cpu().numpy(),
                    conns=best_conns.detach().cpu().numpy(),
                    fitness=self.best_fitness,
                )

        return state, self.best_genome

    def analysis(self, state, pop, fitnesses):
        generation = int(state.generation)
        valid_fitnesses = fitnesses[~np.isinf(fitnesses)]
        if len(valid_fitnesses) == 0:
            max_f = min_f = mean_f = std_f = float("nan")
        else:
            max_f = float(np.max(valid_fitnesses))
            min_f = float(np.min(valid_fitnesses))
            mean_f = float(np.mean(valid_fitnesses))
            std_f = float(np.std(valid_fitnesses))

        cost_time = time.time() - self.generation_timestamp
        max_idx = int(np.argmax(fitnesses))
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = float(fitnesses[max_idx])
            self.best_genome = (
                pop[0][max_idx].detach().clone(),
                pop[1][max_idx].detach().clone(),
            )

        if self.is_save:
            with open(os.path.join(self.genome_dir, f"{generation}.npz"), "wb") as handle:
                np.savez(
                    handle,
                    nodes=pop[0][max_idx].detach().cpu().numpy(),
                    conns=pop[1][max_idx].detach().cpu().numpy(),
                    fitness=self.best_fitness,
                )
            with open(os.path.join(self.save_dir, "log.txt"), "a", encoding="ascii") as handle:
                handle.write(f"{generation},{max_f},{min_f},{mean_f},{std_f},{cost_time}\n")

        print(
            f"Generation: {generation}, Cost time: {cost_time * 1000:.2f}ms\n"
            f"\tfitness: valid cnt: {len(valid_fitnesses)}, max: {max_f:.4f}, min: {min_f:.4f}, mean: {mean_f:.4f}, std: {std_f:.4f}\n"
        )
        self.algorithm.show_details(state, torch.as_tensor(fitnesses, dtype=torch.float32))

        if self.show_problem_details:
            pop_transformed = self._last_pop_transformed
            if pop_transformed is None:
                pop_transformed = [self.algorithm.transform(state, (pop[0][idx], pop[1][idx])) for idx in range(self.pop_size)]
            self.problem.show_details(state, state.randkey, self.algorithm.forward, pop_transformed)

    def show(self, state, best, *args, **kwargs):
        transformed = self.algorithm.transform(state, best)
        return self.problem.show(state, state.randkey, self.algorithm.forward, transformed, *args, **kwargs)

    def _evaluate_population(self, state, pop_transformed, seed):
        if self.problem.jitable and self._batched_eval_available:
            return self._evaluate_population_batched(state, pop_transformed, seed)

        fitness_device = state.pop_nodes.device if "pop_nodes" in state else None
        fitnesses = []
        for idx, transformed in enumerate(pop_transformed):
            individual_seed = seed + idx
            fitness = self.problem.evaluate(state, individual_seed, self.algorithm.forward, transformed)
            fitnesses.append(torch.as_tensor(fitness, dtype=torch.float32, device=fitness_device))
        return torch.stack(fitnesses)

    def _evaluate_population_batched(self, state, pop_transformed, seed):
        batch_size = self.eval_batch_size or self.pop_size
        fitness_device = state.pop_nodes.device if "pop_nodes" in state else None
        fitness_batches = []
        batched_evaluator = self._get_batched_evaluator()

        for batch_start in range(0, self.pop_size, batch_size):
            batch_end = min(batch_start + batch_size, self.pop_size)
            batch_transformed = self._stack_tree(pop_transformed[batch_start:batch_end])
            batch_seeds = torch.arange(batch_start, batch_end, dtype=torch.int64, device=seed.device) + seed
            try:
                batch_fitnesses = batched_evaluator(state, batch_seeds, batch_transformed)
            except RuntimeError:
                self._batched_eval_available = False
                return self._evaluate_population(state, pop_transformed, seed)
            fitness_batches.append(torch.as_tensor(batch_fitnesses, dtype=torch.float32, device=fitness_device))

        return torch.cat(fitness_batches, dim=0)

    def _get_batched_evaluator(self):
        if self._compiled_batched_evaluator is None:
            def batched_evaluator(eval_state, batch_seeds, batch_transformed):
                single_evaluator = lambda individual_seed, individual_transformed: self.problem.evaluate(
                    eval_state,
                    individual_seed,
                    self.algorithm.forward,
                    individual_transformed,
                )
                return torch.vmap(single_evaluator)(batch_seeds, batch_transformed)

            self._compiled_batched_evaluator = batched_evaluator

        return self._compiled_batched_evaluator

    def _stack_tree(self, values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(values)
        if isinstance(first, tuple):
            return tuple(self._stack_tree([value[idx] for value in values]) for idx in range(len(first)))
        if isinstance(first, list):
            return [self._stack_tree([value[idx] for value in values]) for idx in range(len(first))]
        if isinstance(first, Sequence):
            return type(first)(self._stack_tree([value[idx] for value in values]) for idx in range(len(first)))
        raise TypeError(f"Unsupported transformed value type for batching: {type(first)!r}")