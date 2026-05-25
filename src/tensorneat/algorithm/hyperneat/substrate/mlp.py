from typing import List, Tuple

import numpy as np

from .default import DefaultSubstrate


class MLPSubstrate(DefaultSubstrate):
    connection_type = "feedforward"

    def __init__(self, layers: List[int], coor_range: Tuple[float, float, float, float] = (-1, 1, -1, 1)):
        if len(layers) < 2:
            raise ValueError("The number of layers should be at least 2")
        if any(layer <= 0 for layer in layers):
            raise ValueError("Each layer size must be positive")
        if not (coor_range[0] < coor_range[1] and coor_range[2] < coor_range[3]):
            raise ValueError("Invalid coordinate range")

        query_coors, nodes, conns = analysis_substrate(layers, coor_range)
        super().__init__(layers[0], layers[-1], query_coors, nodes, conns)


def analysis_substrate(layers, coor_range):
    x_min, x_max, y_min, y_max = coor_range
    layer_cnt = len(layers)
    y_interval = (y_max - y_min) / (layer_cnt - 1)

    input_indices = list(range(layers[0]))
    input_coors = cal_coors(layers[0], x_min, x_max, y_min)
    output_indices = list(range(layers[0], layers[0] + layers[-1]))
    output_coors = cal_coors(layers[-1], x_min, x_max, y_max)

    if layer_cnt == 2:
        node_layers = [input_indices, output_indices]
        node_coors = [*input_coors, *output_coors]
    else:
        hidden_idx = layers[0] + layers[-1]
        hidden_layers = []
        hidden_coors = []
        for layer_idx in range(1, layer_cnt - 1):
            y_coor = y_min + layer_idx * y_interval
            indices = list(range(hidden_idx, hidden_idx + layers[layer_idx]))
            coors = cal_coors(layers[layer_idx], x_min, x_max, y_coor)
            hidden_layers.append(indices)
            hidden_coors.extend(coors)
            hidden_idx += layers[layer_idx]
        node_layers = [input_indices, *hidden_layers, output_indices]
        node_coors = [*input_coors, *output_coors, *hidden_coors]

    query_coors = []
    correspond_keys = []
    for layer_idx in range(layer_cnt - 1):
        for i in range(layers[layer_idx]):
            for j in range(layers[layer_idx + 1]):
                neuron1 = node_layers[layer_idx][i]
                neuron2 = node_layers[layer_idx + 1][j]
                query_coors.append((*node_coors[neuron1], *node_coors[neuron2]))
                correspond_keys.append((neuron1, neuron2))

    ordered_nodes = [*node_layers[0], *node_layers[-1]]
    for layer in node_layers[1:-1]:
        ordered_nodes.extend(layer)

    nodes = np.array(ordered_nodes, dtype=np.float32)[:, None]
    conns = np.zeros((len(correspond_keys), 3), dtype=np.float32)
    conns[:, :2] = np.array(correspond_keys, dtype=np.float32)
    query_coors = np.array(query_coors, dtype=np.float32)
    return query_coors, nodes, conns


def cal_coors(neuron_cnt, x_min, x_max, y_coor):
    if neuron_cnt == 1:
        return [((x_min + x_max) / 2, y_coor)]
    x_interval = (x_max - x_min) / (neuron_cnt - 1)
    return [(x_min + x_interval * i, y_coor) for i in range(neuron_cnt)]