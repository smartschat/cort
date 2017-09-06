from cort.coreference.approaches import hypergraph


__author__ = 'martscsn'


class HypergraphPairCost(hypergraph.HypergraphPerceptron):
    def _get_cost(self, ana, cluster, arc_information):
        return self.cost_scaling*arc_information[(ana, cluster[0])][1][0]


class HypergraphHyperCost(hypergraph.HypergraphPerceptron):
    def _get_cost(self, ana, cluster, arc_information):
        return sum([self.cost_scaling*arc_information[(ana, m)][1][0] for m in cluster])

