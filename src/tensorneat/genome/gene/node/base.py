from .. import BaseGene


class BaseNode(BaseGene):
    "Base class for node genes."

    fixed_attrs = ["index"]

    def __init__(self):
        super().__init__()

    def forward(self, state, attrs, inputs, is_output_node=False, valid_mask=None):
        raise NotImplementedError

    def repr(self, state, node, precision=2, idx_width=3, func_width=8):
        idx = int(node[0])
        return "{}(idx={:<{idx_width}})".format(
            self.__class__.__name__, idx, idx_width=idx_width
        )

    def to_dict(self, state, node):
        return {"idx": int(node[0])}

    def sympy_func(self, state, node_dict, inputs, is_output_node=False):
        raise NotImplementedError