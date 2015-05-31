__author__ = 'smartschat'


class CorefMultigraphCreator:
    def __init__(self,
                 positive_features,
                 negative_features,
                 weighting_function,
                 relation_weights,
                 construct_when_negative=False):
        self.positive_features = positive_features
        self.negative_features = negative_features
        self.weighting_function = weighting_function
        self.relation_weights = relation_weights
        self.construct_when_negative = construct_when_negative

    def construct_graph_from_mentions(self, mentions):
        nodes = []
        edges = {}

        for i in range(0, len(mentions)):
            anaphor = mentions[i]

            nodes.append(anaphor)

            edges[anaphor] = self.construct_for_one_mention(mentions, i)

        return CorefMultigraph(nodes,
                               edges,
                               self.weighting_function,
                               self.relation_weights)

    def construct_for_one_mention(self, mentions, i):
        anaphor = mentions[i]

        edges = {}

        # do not include dummy mention
        for j in range(i-1, 0, -1):
            antecedent = mentions[j]
            if self.construct_when_negative:
                edges[antecedent] = self.get_edge_relations(anaphor, antecedent)
            else:
                if not self.has_negative(anaphor, antecedent):
                    edges[antecedent] = {
                        "negative_relations": [],
                        "positive_relations": self.get_positive_relations(
                            anaphor, antecedent)
                    }

        return edges

    def get_edge_relations(self, anaphor, antecedent):
        relations = {
            "negative_relations":
            self.get_negative_relations(anaphor, antecedent),
            "positive_relations":
            self.get_positive_relations(anaphor, antecedent)
        }

        return relations

    def has_negative(self, anaphor, antecedent):
        for r in self.negative_features:
            if r(anaphor, antecedent):
                return True

    def get_negative_relations(self, anaphor, antecedent):
        negative_relations = []

        for r in self.negative_features:
            if r(anaphor, antecedent):
                negative_relations.append(r)

        return negative_relations

    def get_positive_relations(self, anaphor, antecedent):
        positive_relations = []

        for r in self.positive_features:
            if r(anaphor, antecedent):
                positive_relations.append(r)

        return positive_relations


class CorefMultigraph:
    def __init__(self, nodes, edges, weighting_function, relation_weights):
        self.nodes = nodes
        self.edges = edges
        self.weighting_function = weighting_function
        self.relation_weights = relation_weights

    def get_weight(self, anaphor, antecedent):
        return self.weighting_function(
            anaphor,
            antecedent,
            self.edges[anaphor][antecedent],
            self.relation_weights)
