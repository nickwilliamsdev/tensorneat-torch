from typing import List, Set, Tuple, Union

import torch

from .tools import I_INF, fetch_first


def topological_sort(nodes, conns):
    mask = torch.isnan(nodes[:, 0])
    in_degree = torch.where(mask, torch.nan, torch.sum(conns, dim=0).to(dtype=nodes.dtype))
    result = torch.full(in_degree.shape, I_INF, dtype=torch.int64, device=nodes.device)

    all_indices = torch.arange(conns.shape[0], device=nodes.device, dtype=torch.int64)
    for idx in range(result.shape[0]):
        node_idx = fetch_first(in_degree == 0.0)
        has_candidate = node_idx != I_INF
        safe_node_idx = torch.where(has_candidate, node_idx, torch.zeros_like(node_idx))

        result[idx] = torch.where(has_candidate, safe_node_idx, result[idx])

        selected_mask = all_indices == safe_node_idx
        marked_in_degree = torch.where(selected_mask, torch.full_like(in_degree, -1), in_degree)
        children = conns[safe_node_idx, :].to(dtype=torch.bool)
        reduced_in_degree = torch.where(children, marked_in_degree - 1, marked_in_degree)
        in_degree = torch.where(has_candidate, reduced_in_degree, in_degree)

    return result


def topological_sort_python(
    nodes: Union[Set[int], List[int]],
    conns: Union[Set[Tuple[int, int]], List[Tuple[int, int]]],
) -> Tuple[List[int], List[List[int]]]:
    nodes = nodes.copy()
    conns = conns.copy()

    in_degree = {node: 0 for node in nodes}
    for conn in conns:
        in_degree[conn[1]] += 1

    topo_order = []
    topo_layer = []
    zero_in_degree_nodes = [node for node in nodes if in_degree[node] == 0]

    while zero_in_degree_nodes:
        for node in zero_in_degree_nodes:
            nodes.remove(node)

        zero_in_degree_nodes = sorted(zero_in_degree_nodes)
        topo_layer.append(zero_in_degree_nodes.copy())

        for node in zero_in_degree_nodes:
            topo_order.append(node)
            for conn in list(conns):
                if conn[0] == node:
                    in_degree[conn[1]] -= 1
                    conns.remove(conn)

        zero_in_degree_nodes = [node for node in nodes if in_degree[node] == 0]

    if conns or nodes:
        raise ValueError("Graph has at least one cycle, topological sort not possible")

    return topo_order, topo_layer


def find_useful_nodes(
    nodes: Union[Set[int], List[int]],
    conns: Union[Set[Tuple[int, int]], List[Tuple[int, int]]],
    output_idx: Set[int],
) -> Set[int]:
    useful_nodes = set(output_idx)
    while True:
        additions = set()
        for in_idx, out_idx in conns:
            if out_idx in useful_nodes and in_idx not in useful_nodes:
                additions.add(in_idx)
        if not additions:
            break
        useful_nodes |= additions
    return useful_nodes


def check_cycles(nodes, conns, from_idx, to_idx):
    del nodes

    conns = conns.clone()
    conns[from_idx, to_idx] = True

    frontier = torch.zeros(conns.shape[0], dtype=torch.bool, device=conns.device)
    frontier[to_idx] = True
    done = torch.tensor(False, dtype=torch.bool, device=conns.device)
    conns_float = conns.to(dtype=torch.float32)

    for _ in range(conns.shape[0]):
        reachable = frontier.to(dtype=torch.float32) @ conns_float
        next_frontier = torch.logical_or(frontier, reachable > 0)
        next_done = torch.logical_or(done, torch.logical_or(torch.all(next_frontier == frontier), next_frontier[from_idx]))
        frontier = torch.where(done, frontier, next_frontier)
        done = next_done

    return frontier[from_idx]