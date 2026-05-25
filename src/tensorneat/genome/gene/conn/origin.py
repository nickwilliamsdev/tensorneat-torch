import torch

from .default import DefaultConn


class OriginConn(DefaultConn):
    """
    Implementation of connections in the original NEAT paper.
    """

    fixed_attrs = ["input_index", "output_index", "historical_marker"]
    custom_attrs = ["weight"]

    def crossover(self, state, generator, attrs1, attrs2):
        del state
        pick_first = torch.randn((), generator=generator, device=attrs1.device) > 0
        return attrs1 if bool(pick_first) else attrs2

    def get_historical_marker(self, state, gene_array):
        del state
        return gene_array[2]

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        del state, func_width
        in_idx, out_idx, historical_marker, weight = conn
        return "{}(in: {:<{idx_width}}, out: {:<{idx_width}}, historical_marker: {:<{idx_width}}, weight: {:<{float_width}})".format(
            self.__class__.__name__,
            int(in_idx),
            int(out_idx),
            int(historical_marker),
            round(float(weight), precision),
            idx_width=idx_width,
            float_width=precision + 3,
        )

    def to_dict(self, state, conn):
        del state
        return {
            "in": int(conn[0]),
            "out": int(conn[1]),
            "historical_marker": int(conn[2]),
            "weight": torch.tensor(float(conn[3]), dtype=torch.float32),
        }