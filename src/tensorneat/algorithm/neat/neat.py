from typing import Callable

import torch

from .species import SpeciesController
from ..base import BaseAlgorithm
from tensorneat.common import State
from tensorneat.genome import BaseGenome


def _state_seed(state, default=0):
    value = state.state_dict.get("randkey", torch.tensor(default, dtype=torch.int64))
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


class NEAT(BaseAlgorithm):
    def __init__(
        self,
        genome: BaseGenome,
        pop_size: int,
        species_size: int = 10,
        max_stagnation: int = 15,
        species_elitism: int = 2,
        spawn_number_change_rate: float = 0.5,
        genome_elitism: int = 2,
        survival_threshold: float = 0.1,
        min_species_size: int = 1,
        compatibility_threshold: float = 2.0,
        species_fitness_func: Callable = torch.max,
        species_number_calculate_by: str = "rank",
    ):
        if species_number_calculate_by not in ["rank", "fitness"]:
            raise ValueError("species_number_calculate_by should be either 'rank' or 'fitness'")

        self.genome = genome
        self.pop_size = pop_size
        self.species_controller = SpeciesController(
            pop_size,
            species_size,
            max_stagnation,
            species_elitism,
            spawn_number_change_rate,
            genome_elitism,
            survival_threshold,
            min_species_size,
            compatibility_threshold,
            species_fitness_func,
            species_number_calculate_by,
        )

    def setup(self, state=State()):
        state = self.genome.setup(state)
        seed = _state_seed(state)
        pop_nodes = []
        pop_conns = []
        for idx in range(self.pop_size):
            generator = _generator(seed + idx)
            nodes, conns = self.genome.initialize(state, generator)
            pop_nodes.append(nodes)
            pop_conns.append(conns)

        pop_nodes = torch.stack(pop_nodes)
        pop_conns = torch.stack(pop_conns)
        state = state.register(
            pop_nodes=pop_nodes,
            pop_conns=pop_conns,
            generation=torch.tensor(0.0, dtype=torch.float32),
        )
        state = self.species_controller.setup(state, pop_nodes[0], pop_conns[0])
        state = self.species_controller.speciate(state, self.genome.execute_distance)
        return state.update(randkey=torch.tensor(seed + self.pop_size + 1, dtype=torch.int64))

    def ask(self, state):
        return state.pop_nodes, state.pop_conns

    def tell(self, state, fitness):
        state = state.update(generation=state.generation + 1)
        state, winner, loser, elite_mask = self.species_controller.update_species(state, fitness)
        state = self._create_next_generation(state, winner, loser, elite_mask)
        state = self.species_controller.speciate(state, self.genome.execute_distance)
        return state

    def transform(self, state, individual):
        nodes, conns = individual
        return self.genome.transform(state, nodes, conns)

    def forward(self, state, transformed, inputs):
        return self.genome.forward(state, transformed, inputs)

    @property
    def num_inputs(self):
        return self.genome.num_inputs

    @property
    def num_outputs(self):
        return self.genome.num_outputs

    def _create_next_generation(self, state, winner, loser, elite_mask):
        all_nodes_keys = state.pop_nodes[:, :, 0]
        valid_node_keys = all_nodes_keys[~torch.isnan(all_nodes_keys)]
        max_node_key = int(torch.max(valid_node_keys).item()) if valid_node_keys.numel() else 0
        new_node_keys = torch.arange(self.pop_size, dtype=torch.float32) + (max_node_key + 1)

        if "historical_marker" in self.genome.conn_gene.fixed_attrs:
            markers = []
            for pop_idx in range(self.pop_size):
                for conn in state.pop_conns[pop_idx]:
                    if not torch.isnan(conn[0]):
                        markers.append(float(self.genome.conn_gene.get_historical_marker(state, conn)))
            max_conn_marker = max(markers) if markers else 0.0
            new_conn_markers = torch.arange(self.pop_size * 3, dtype=torch.float32).reshape(self.pop_size, 3) + (max_conn_marker + 1)
        else:
            new_conn_markers = torch.zeros((self.pop_size, 3), dtype=torch.float32)

        seed = _state_seed(state)
        new_nodes = []
        new_conns = []
        for idx in range(self.pop_size):
            crossover_gen = _generator(seed + idx)
            mutate_gen = _generator(seed + self.pop_size + idx)
            wpn = state.pop_nodes[int(winner[idx])]
            wpc = state.pop_conns[int(winner[idx])]
            lpn = state.pop_nodes[int(loser[idx])]
            lpc = state.pop_conns[int(loser[idx])]
            child_nodes, child_conns = self.genome.execute_crossover(state, crossover_gen, wpn, wpc, lpn, lpc)
            mutated_nodes, mutated_conns = self.genome.execute_mutation(
                state,
                mutate_gen,
                child_nodes,
                child_conns,
                new_node_keys[idx],
                new_conn_markers[idx],
            )
            if bool(elite_mask[idx]):
                new_nodes.append(child_nodes)
                new_conns.append(child_conns)
            else:
                new_nodes.append(mutated_nodes)
                new_conns.append(mutated_conns)

        return state.update(
            randkey=torch.tensor(seed + self.pop_size * 2 + 1, dtype=torch.int64),
            pop_nodes=torch.stack(new_nodes),
            pop_conns=torch.stack(new_conns),
        )

    def show_details(self, state, fitness):
        member_count = state.species.member_count.detach().cpu()
        species_sizes = [int(value) for value in member_count if value > 0]
        pop_nodes = state.pop_nodes.detach().cpu()
        pop_conns = state.pop_conns.detach().cpu()
        nodes_cnt = (~torch.isnan(pop_nodes[:, :, 0])).sum(dim=1)
        conns_cnt = (~torch.isnan(pop_conns[:, :, 0])).sum(dim=1)

        print(
            f"\tnode counts: max: {int(nodes_cnt.max())}, min: {int(nodes_cnt.min())}, mean: {float(nodes_cnt.float().mean()):.2f}\n"
            f"\tconn counts: max: {int(conns_cnt.max())}, min: {int(conns_cnt.min())}, mean: {float(conns_cnt.float().mean()):.2f}\n"
            f"\tspecies: {len(species_sizes)}, {species_sizes}\n"
        )