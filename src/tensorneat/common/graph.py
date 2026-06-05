from typing import List, Set, Tuple, Union

import torch

from .tools import I_INF, fetch_first


def topological_sort(nodes, conns):
    mask = torch.isnan(nodes[:, 0])
    in_degree = torch.where(mask, torch.nan, torch.sum(conns, dim=0).to(dtype=nodes.dtype))
    result = torch.full(in_degree.shape, I_INF, dtype=torch.int64, device=nodes.device)

    idx = 0
    while True:
        node_idx = fetch_first(in_degree == 0.0)
        if int(node_idx) == I_INF:
            break

        result[idx] = node_idx
        in_degree[node_idx] = -1

        children = conns[node_idx, :].to(dtype=torch.bool)
        in_degree = torch.where(children, in_degree - 1, in_degree)
        idx += 1

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

    visited = torch.zeros(conns.shape[0], dtype=torch.bool, device=conns.device)
    frontier = visited.clone()
    frontier[to_idx] = True

    while True:
        if torch.equal(visited, frontier) or bool(frontier[from_idx]):
            break
        visited = frontier
        reachable = visited.to(dtype=torch.float32) @ conns.to(dtype=torch.float32)
        frontier = torch.logical_or(visited, reachable > 0)

    return frontier[from_idx]