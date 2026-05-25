from tensorneat.common import StatefulBaseClass


class BaseCrossover(StatefulBaseClass):
    def __call__(self, state, genome, generator, nodes1, nodes2, conns1, conns2):
        raise NotImplementedError