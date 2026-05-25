import torch

from .default import DefaultNode


class OriginNode(DefaultNode):
    """
    Implementation of nodes in the original NEAT paper.
    """

    def crossover(self, state, generator, attrs1, attrs2):
        del state
        pick_first = torch.randn((), generator=generator, device=attrs1.device) > 0
        return attrs1 if bool(pick_first) else attrs2