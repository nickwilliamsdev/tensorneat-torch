import torch

from tensorneat.common import StatefulBaseClass, hash_array


class BaseGene(StatefulBaseClass):
    "Base class for node genes or connection genes."

    fixed_attrs = []
    custom_attrs = []

    def __init__(self):
        super().__init__()

    def new_identity_attrs(self, state):
        raise NotImplementedError

    def new_random_attrs(self, state, generator):
        raise NotImplementedError

    def mutate(self, state, generator, attrs):
        raise NotImplementedError

    def crossover(self, state, generator, attrs1, attrs2):
        mask = torch.randn(
            attrs1.shape,
            generator=generator,
            device=attrs1.device,
            dtype=attrs1.dtype,
        ) > 0
        return torch.where(mask, attrs1, attrs2)

    def distance(self, state, attrs1, attrs2):
        raise NotImplementedError

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    @property
    def length(self):
        return len(self.fixed_attrs) + len(self.custom_attrs)

    def repr(self, state, gene, precision=2):
        raise NotImplementedError

    def hash(self, gene):
        return hash_array(gene)