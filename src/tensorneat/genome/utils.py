import numpy as np
import torch

from .gene import BaseGene
from tensorneat.common import I_INF, fetch_first


def unflatten_conns(nodes, conns):
    node_keys = nodes[:, 0]
    size = nodes.shape[0]
    unflatten = torch.full((size, size), I_INF, dtype=torch.int64, device=nodes.device)

    key_to_index = {}
    for idx, key in enumerate(node_keys):
        if not torch.isnan(key):
            key_to_index[int(key.item())] = idx

    for conn_idx, conn in enumerate(conns):
        in_key, out_key = conn[:2]
        if torch.isnan(in_key) or torch.isnan(out_key):
            continue
        in_idx = key_to_index.get(int(in_key.item()))
        out_idx = key_to_index.get(int(out_key.item()))
        if in_idx is not None and out_idx is not None:
            unflatten[in_idx, out_idx] = conn_idx

    return unflatten


def valid_cnt(nodes_or_conns):
    return int(torch.sum(~torch.isnan(nodes_or_conns[:, 0])).item())


def extract_gene_attrs(gene: BaseGene, gene_array):
    return gene_array[len(gene.fixed_attrs) :]


def set_gene_attrs(gene: BaseGene, gene_array, attrs):
    updated = gene_array.clone()
    updated[len(gene.fixed_attrs) :] = attrs
    return updated


def add_node(nodes, fix_attrs, custom_attrs):
    pos = int(fetch_first(torch.isnan(nodes[:, 0])).item())
    updated = nodes.clone()
    updated[pos] = torch.cat((fix_attrs, custom_attrs))
    return updated


def delete_node_by_pos(nodes, pos):
    updated = nodes.clone()
    updated[pos] = torch.nan
    return updated


def add_conn(conns, fix_attrs, custom_attrs):
    pos = int(fetch_first(torch.isnan(conns[:, 0])).item())
    updated = conns.clone()
    updated[pos] = torch.cat((fix_attrs, custom_attrs))
    return updated


def delete_conn_by_pos(conns, pos):
    updated = conns.clone()
    updated[pos] = torch.nan
    return updated


def re_cound_idx(nodes, conns, input_idx, output_idx):
    nodes_np = nodes.detach().cpu().numpy().copy()
    conns_np = conns.detach().cpu().numpy().copy()

    next_key = max(*input_idx, *output_idx) + 1
    old2new = {}
    fixed_keys = set(input_idx + output_idx)

    for key in nodes_np[:, 0]:
        if np.isnan(key):
            continue
        int_key = int(key)
        if int_key in fixed_keys:
            continue
        if int_key not in old2new:
            old2new[int_key] = next_key
            next_key += 1

    for idx, key in enumerate(nodes_np[:, 0]):
        if not np.isnan(key) and int(key) in old2new:
            nodes_np[idx, 0] = old2new[int(key)]

    for idx, (in_key, out_key) in enumerate(conns_np[:, :2]):
        if not np.isnan(in_key) and int(in_key) in old2new:
            conns_np[idx, 0] = old2new[int(in_key)]
        if not np.isnan(out_key) and int(out_key) in old2new:
            conns_np[idx, 1] = old2new[int(out_key)]

    return (
        torch.from_numpy(nodes_np).to(device=nodes.device, dtype=nodes.dtype),
        torch.from_numpy(conns_np).to(device=conns.device, dtype=conns.dtype),
    )