from typing import Callable

import torch

from tensorneat.common import State, StatefulBaseClass, argmin_with_mask, fetch_first, rank_elements


def _as_seed(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _generator(seed, device=None):
    generator = torch.Generator(device=device or "cpu")
    generator.manual_seed(_as_seed(seed))
    return generator


class SpeciesController(StatefulBaseClass):
    def __init__(
        self,
        pop_size,
        species_size,
        max_stagnation,
        species_elitism,
        spawn_number_change_rate,
        genome_elitism,
        survival_threshold,
        min_species_size,
        compatibility_threshold,
        species_fitness_func: Callable,
        species_number_calculate_by,
    ):
        self.pop_size = pop_size
        self.species_size = species_size
        self.max_stagnation = max_stagnation
        self.species_elitism = species_elitism
        self.spawn_number_change_rate = spawn_number_change_rate
        self.genome_elitism = genome_elitism
        self.survival_threshold = survival_threshold
        self.min_species_size = min_species_size
        self.compatibility_threshold = compatibility_threshold
        self.species_fitness_func = species_fitness_func
        self.species_number_calculate_by = species_number_calculate_by

    def setup(self, state, first_nodes, first_conns):
        device = first_nodes.device
        species_keys = torch.full((self.species_size,), torch.nan, dtype=torch.float32, device=device)
        best_fitness = torch.full((self.species_size,), torch.nan, dtype=torch.float32, device=device)
        last_improved = torch.full((self.species_size,), torch.nan, dtype=torch.float32, device=device)
        member_count = torch.full((self.species_size,), torch.nan, dtype=torch.float32, device=device)
        idx2species = torch.zeros(self.pop_size, dtype=torch.float32, device=device)
        center_nodes = torch.full((self.species_size, *first_nodes.shape), torch.nan, dtype=first_nodes.dtype, device=device)
        center_conns = torch.full((self.species_size, *first_conns.shape), torch.nan, dtype=first_conns.dtype, device=device)

        species_keys[0] = 0
        best_fitness[0] = -torch.inf
        last_improved[0] = 0
        member_count[0] = self.pop_size
        center_nodes[0] = first_nodes
        center_conns[0] = first_conns

        species_state = State(
            species_keys=species_keys,
            best_fitness=best_fitness,
            last_improved=last_improved,
            member_count=member_count,
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_conns=center_conns,
            next_species_key=torch.tensor(1.0, dtype=torch.float32, device=device),
        )
        return state.register(species=species_state)

    def update_species(self, state, fitness):
        species_state = state.species
        species_fitness = self._update_species_fitness(species_state, fitness)
        species_state, species_fitness = self._stagnation(species_state, species_fitness, state.generation)

        sortable = torch.where(torch.isnan(species_fitness), torch.full_like(species_fitness, -torch.inf), species_fitness)
        sort_indices = torch.argsort(sortable, descending=True)
        species_state = species_state.update(
            species_keys=species_state.species_keys[sort_indices],
            best_fitness=species_state.best_fitness[sort_indices],
            last_improved=species_state.last_improved[sort_indices],
            member_count=species_state.member_count[sort_indices],
            center_nodes=species_state.center_nodes[sort_indices],
            center_conns=species_state.center_conns[sort_indices],
        )
        species_fitness = species_fitness[sort_indices]

        if self.species_number_calculate_by == "rank":
            spawn_number = self._cal_spawn_numbers_by_rank(species_state)
        else:
            spawn_number = self._cal_spawn_numbers_by_fitness(species_state)

        state_seed = state.state_dict.get("randkey", torch.tensor(0, dtype=torch.int64))
        winner, loser, elite_mask = self._create_crossover_pair(
            species_state,
            state_seed,
            spawn_number,
            fitness,
        )

        next_seed = torch.tensor(_as_seed(state_seed) + 1, dtype=torch.int64)
        return state.update(randkey=next_seed, species=species_state), winner, loser, elite_mask

    def _update_species_fitness(self, species_state, fitness):
        values = []
        for idx in range(self.species_size):
            key = species_state.species_keys[idx]
            if torch.isnan(key):
                values.append(torch.tensor(torch.nan, dtype=fitness.dtype, device=fitness.device))
                continue
            mask = species_state.idx2species == key
            if not bool(torch.any(mask)):
                values.append(torch.tensor(torch.nan, dtype=fitness.dtype, device=fitness.device))
                continue
            selected = fitness[mask]
            values.append(self.species_fitness_func(selected))
        return torch.stack(values)

    def _stagnation(self, species_state, species_fitness, generation):
        best_fitness = species_state.best_fitness.clone()
        last_improved = species_state.last_improved.clone()
        stagnation = torch.zeros(self.species_size, dtype=torch.bool, device=species_fitness.device)

        for idx in range(self.species_size):
            fit = species_fitness[idx]
            if torch.isnan(fit) or torch.isnan(species_state.species_keys[idx]):
                stagnation[idx] = True
                continue
            improved = torch.isnan(best_fitness[idx]) or fit > best_fitness[idx]
            if bool(improved):
                best_fitness[idx] = fit
                last_improved[idx] = generation
            elif generation - last_improved[idx] > self.max_stagnation:
                stagnation[idx] = True

        fitness_for_rank = torch.where(torch.isnan(species_fitness), torch.full_like(species_fitness, -torch.inf), species_fitness)
        species_rank = rank_elements(fitness_for_rank)
        stagnation = torch.where(species_rank < self.species_elitism, torch.tensor(False, device=stagnation.device), stagnation)

        species_keys = species_state.species_keys.clone()
        member_count = species_state.member_count.clone()
        center_nodes = species_state.center_nodes.clone()
        center_conns = species_state.center_conns.clone()
        species_fitness = species_fitness.clone()

        for idx in range(self.species_size):
            if bool(stagnation[idx]):
                species_keys[idx] = torch.nan
                best_fitness[idx] = torch.nan
                last_improved[idx] = torch.nan
                member_count[idx] = torch.nan
                center_nodes[idx] = torch.nan
                center_conns[idx] = torch.nan
                species_fitness[idx] = torch.nan

        return species_state.update(
            species_keys=species_keys,
            best_fitness=best_fitness,
            last_improved=last_improved,
            member_count=member_count,
            center_nodes=center_nodes,
            center_conns=center_conns,
        ), species_fitness

    def _cal_spawn_numbers_by_rank(self, species_state):
        valid = ~torch.isnan(species_state.species_keys)
        valid_species_num = int(torch.sum(valid).item())
        if valid_species_num == 0:
            result = torch.zeros(self.species_size, dtype=torch.int32, device=species_state.species_keys.device)
            result[0] = self.pop_size
            return result

        denominator = (valid_species_num + 1) * valid_species_num / 2
        rank_score = torch.clamp(valid_species_num - torch.arange(self.species_size, device=valid.device), min=0)
        spawn_rate = rank_score / denominator
        target = torch.floor(spawn_rate * self.pop_size)
        previous = torch.nan_to_num(species_state.member_count, nan=0.0)
        spawn_number = previous + (target - previous) * self.spawn_number_change_rate
        spawn_number = torch.where(
            valid & (spawn_number < self.min_species_size),
            torch.tensor(float(self.min_species_size), device=spawn_number.device),
            spawn_number,
        )
        spawn_number = torch.where(valid, spawn_number, torch.tensor(0.0, device=spawn_number.device))
        spawn_number = spawn_number.to(dtype=torch.int32)
        error = self.pop_size - int(torch.sum(spawn_number).item())
        spawn_number[0] += error
        return spawn_number

    def _cal_spawn_numbers_by_fitness(self, species_state):
        species_fitness = species_state.best_fitness.clone()
        valid = ~torch.isnan(species_fitness)
        if not bool(torch.any(valid)):
            result = torch.zeros(self.species_size, dtype=torch.int32, device=species_fitness.device)
            result[0] = self.pop_size
            return result

        min_fitness = torch.min(species_fitness[valid])
        shifted = torch.where(valid, species_fitness - min_fitness + 1, torch.tensor(0.0, device=species_fitness.device))
        spawn_rate = shifted / torch.sum(shifted)
        target = torch.floor(spawn_rate * self.pop_size)
        previous = torch.nan_to_num(species_state.member_count, nan=0.0)
        spawn_number = previous + (target - previous) * self.spawn_number_change_rate
        spawn_number = torch.where(
            valid & (spawn_number < self.min_species_size),
            torch.tensor(float(self.min_species_size), device=spawn_number.device),
            spawn_number,
        )
        spawn_number = torch.where(valid, spawn_number, torch.tensor(0.0, device=spawn_number.device))
        spawn_number = spawn_number.to(dtype=torch.int32)
        error = self.pop_size - int(torch.sum(spawn_number).item())
        spawn_number[0] += error
        return spawn_number

    def _create_crossover_pair(self, species_state, seed, spawn_number, fitness):
        device = species_state.idx2species.device
        generator = _generator(seed, device.type)
        winner = torch.zeros(self.pop_size, dtype=torch.long, device=device)
        loser = torch.zeros(self.pop_size, dtype=torch.long, device=device)
        elite_mask = torch.zeros(self.pop_size, dtype=torch.bool, device=device)

        fill_start = 0
        population_indices = torch.arange(self.pop_size, device=device)

        for species_idx in range(self.species_size):
            if torch.isnan(species_state.species_keys[species_idx]):
                continue
            count = int(spawn_number[species_idx].item())
            if count <= 0:
                continue

            members_mask = species_state.idx2species == species_state.species_keys[species_idx]
            members = population_indices[members_mask]
            if members.numel() == 0:
                continue

            member_fitness = fitness[members]
            sorted_order = torch.argsort(member_fitness, descending=True)
            sorted_members = members[sorted_order]
            survive_size = max(1, int(self.survival_threshold * len(sorted_members)))
            pool = sorted_members[:survive_size]

            for local_idx in range(count):
                global_idx = min(fill_start + local_idx, self.pop_size - 1)
                if local_idx < min(self.genome_elitism, len(sorted_members)):
                    parent1 = sorted_members[local_idx]
                    parent2 = sorted_members[local_idx]
                    elite_mask[global_idx] = True
                else:
                    sample = torch.randint(0, len(pool), (2,), generator=generator, device=device)
                    parent1 = pool[int(sample[0])]
                    parent2 = pool[int(sample[1])]

                if fitness[parent1] >= fitness[parent2]:
                    winner[global_idx] = parent1
                    loser[global_idx] = parent2
                else:
                    winner[global_idx] = parent2
                    loser[global_idx] = parent1

            fill_start += count
            if fill_start >= self.pop_size:
                break

        if fill_start < self.pop_size:
            winner[fill_start:] = winner[0]
            loser[fill_start:] = loser[0]
            elite_mask[fill_start:] = False

        return winner, loser, elite_mask

    def speciate(self, state, genome_distance_func: Callable):
        species_state = state.species
        pop_nodes = state.pop_nodes
        pop_conns = state.pop_conns

        device = pop_nodes.device
        idx2species = torch.full((self.pop_size,), torch.nan, dtype=torch.float32, device=device)
        o2c_distances = torch.full((self.pop_size,), torch.inf, dtype=torch.float32, device=device)
        center_nodes = species_state.center_nodes.clone()
        center_conns = species_state.center_conns.clone()
        species_keys = species_state.species_keys.clone()
        next_species_key = float(species_state.next_species_key)

        for i in range(self.species_size):
            if torch.isnan(species_keys[i]):
                continue
            unassigned = torch.isnan(idx2species)
            if not bool(torch.any(unassigned)):
                break
            distances = torch.full((self.pop_size,), torch.inf, dtype=torch.float32, device=device)
            for pop_idx in range(self.pop_size):
                if not bool(unassigned[pop_idx]):
                    continue
                distances[pop_idx] = genome_distance_func(
                    state,
                    center_nodes[i],
                    center_conns[i],
                    pop_nodes[pop_idx],
                    pop_conns[pop_idx],
                )
            closest_idx = int(argmin_with_mask(distances, unassigned).item())
            idx2species[closest_idx] = species_keys[i]
            center_nodes[i] = pop_nodes[closest_idx]
            center_conns[i] = pop_conns[closest_idx]
            o2c_distances[closest_idx] = 0.0

        for pop_idx in range(self.pop_size):
            assigned = False
            for i in range(self.species_size):
                if torch.isnan(species_keys[i]):
                    continue
                distance = genome_distance_func(
                    state,
                    center_nodes[i],
                    center_conns[i],
                    pop_nodes[pop_idx],
                    pop_conns[pop_idx],
                )
                if distance < self.compatibility_threshold and (torch.isnan(idx2species[pop_idx]) or distance < o2c_distances[pop_idx]):
                    idx2species[pop_idx] = species_keys[i]
                    o2c_distances[pop_idx] = distance
                    assigned = True

            if not assigned and bool(torch.isnan(idx2species[pop_idx])):
                new_idx = int(fetch_first(torch.isnan(species_keys)).item())
                if new_idx == int(torch.iinfo(torch.int32).max):
                    fallback = torch.nan_to_num(species_keys[-1], nan=0.0)
                    idx2species[pop_idx] = fallback
                    continue
                species_keys[new_idx] = next_species_key
                center_nodes[new_idx] = pop_nodes[pop_idx]
                center_conns[new_idx] = pop_conns[pop_idx]
                idx2species[pop_idx] = next_species_key
                o2c_distances[pop_idx] = 0.0
                next_species_key += 1

        if bool(torch.any(torch.isnan(idx2species))):
            fallback_mask = ~torch.isnan(species_keys)
            fallback_key = species_keys[int(fetch_first(fallback_mask).item())]
            idx2species = torch.where(torch.isnan(idx2species), fallback_key, idx2species)

        new_created_mask = (~torch.isnan(species_keys)) & torch.isnan(species_state.best_fitness)
        best_fitness = torch.where(new_created_mask, torch.full_like(species_state.best_fitness, -torch.inf), species_state.best_fitness)
        last_improved = torch.where(new_created_mask, torch.full_like(species_state.last_improved, float(state.generation)), species_state.last_improved)
        member_count = torch.full((self.species_size,), torch.nan, dtype=torch.float32, device=device)
        for idx in range(self.species_size):
            if not torch.isnan(species_keys[idx]):
                member_count[idx] = torch.sum(idx2species == species_keys[idx]).to(dtype=torch.float32)

        species_state = species_state.update(
            species_keys=species_keys,
            best_fitness=best_fitness,
            last_improved=last_improved,
            member_count=member_count,
            idx2species=idx2species,
            center_nodes=center_nodes,
            center_conns=center_conns,
            next_species_key=torch.tensor(next_species_key, dtype=torch.float32, device=device),
        )
        return state.update(species=species_state)