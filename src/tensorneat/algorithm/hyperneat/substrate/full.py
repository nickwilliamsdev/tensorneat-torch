import numpy as np

from .default import DefaultSubstrate


class FullSubstrate(DefaultSubstrate):
    connection_type = "recurrent"

    def __init__(
        self,
        input_coors=((-1, -1), (0, -1), (1, -1)),
        hidden_coors=((-1, 0), (0, 0), (1, 0)),
        output_coors=((0, 1),),
    ):
        query_coors, nodes, conns = analysis_substrate(input_coors, output_coors, hidden_coors)
        super().__init__(len(input_coors), len(output_coors), query_coors, nodes, conns)


def analysis_substrate(input_coors, output_coors, hidden_coors):
    input_coors = np.array(input_coors)
    output_coors = np.array(output_coors)
    hidden_coors = np.array(hidden_coors)

    coord_dim = input_coors.shape[1]
    size_i = input_coors.shape[0]
    size_o = output_coors.shape[0]
    size_h = hidden_coors.shape[0]

    input_idx = np.arange(size_i)
    output_idx = np.arange(size_i, size_i + size_o)
    hidden_idx = np.arange(size_i + size_o, size_i + size_o + size_h)

    total_conns = size_i * size_h + size_h * size_h + size_h * size_o
    query_coors = np.zeros((total_conns, coord_dim * 2), dtype=np.float32)
    correspond_keys = np.zeros((total_conns, 2), dtype=np.float32)

    aux_coors, aux_keys = cartesian_product(input_idx, hidden_idx, input_coors, hidden_coors)
    query_coors[0 : size_i * size_h, :] = aux_coors
    correspond_keys[0 : size_i * size_h, :] = aux_keys

    aux_coors, aux_keys = cartesian_product(hidden_idx, hidden_idx, hidden_coors, hidden_coors)
    query_coors[size_i * size_h : size_i * size_h + size_h * size_h, :] = aux_coors
    correspond_keys[size_i * size_h : size_i * size_h + size_h * size_h, :] = aux_keys

    aux_coors, aux_keys = cartesian_product(hidden_idx, output_idx, hidden_coors, output_coors)
    query_coors[size_i * size_h + size_h * size_h :, :] = aux_coors
    correspond_keys[size_i * size_h + size_h * size_h :, :] = aux_keys

    nodes = np.concatenate((input_idx, output_idx, hidden_idx), dtype=np.float32)[:, None]
    conns = np.zeros((correspond_keys.shape[0], 3), dtype=np.float32)
    conns[:, :2] = correspond_keys
    return query_coors, nodes, conns


def cartesian_product(keys1, keys2, coors1, coors2):
    len1 = keys1.shape[0]
    len2 = keys2.shape[0]
    repeated_coors1 = np.repeat(coors1, len2, axis=0)
    repeated_keys1 = np.repeat(keys1, len2)
    tiled_coors2 = np.tile(coors2, (len1, 1))
    tiled_keys2 = np.tile(keys2, len1)
    new_coors = np.concatenate((repeated_coors1, tiled_coors2), axis=1)
    correspond_keys = np.column_stack((repeated_keys1, tiled_keys2))
    return new_coors, correspond_keys