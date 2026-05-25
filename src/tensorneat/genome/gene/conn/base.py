from ..base import BaseGene


class BaseConn(BaseGene):
    "Base class for connection genes."

    fixed_attrs = ["input_index", "output_index"]

    def __init__(self):
        super().__init__()

    def new_zero_attrs(self, state):
        raise NotImplementedError

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        del state, precision, func_width
        in_idx, out_idx = conn[:2]
        return "{}(in: {:<{idx_width}}, out: {:<{idx_width}})".format(
            self.__class__.__name__, int(in_idx), int(out_idx), idx_width=idx_width
        )

    def to_dict(self, state, conn):
        del state
        in_idx, out_idx = conn[:2]
        return {"in": int(in_idx), "out": int(out_idx)}

    def sympy_func(self, state, conn_dict, inputs):
        raise NotImplementedError