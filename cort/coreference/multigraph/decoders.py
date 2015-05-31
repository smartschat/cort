__author__ = 'smartschat'


class MultigraphDecoder:
    def __init__(self, multigraph_creator):
        self.coref_multigraph_creator = multigraph_creator

    def decode(self, corpus):
        for doc in corpus:
            for mention in doc.system_mentions:
                mention.attributes["set_id"] = None

            # discard dummy mention
            self.decode_for_one_document(doc.system_mentions[1:])

    def decode_for_one_document(self, mentions):
        multigraph = \
            self.coref_multigraph_creator.construct_graph_from_mentions(
                mentions)

        for mention in mentions:
            antecedent = self.compute_antecedent(mention, multigraph)

            if antecedent is not None:
                if antecedent.attributes["set_id"] is None:
                    antecedent.attributes["set_id"] = \
                        mentions.index(antecedent)

                mention.attributes["set_id"] = antecedent.attributes["set_id"]
                mention.document.antecedent_decisions[mention.span] = \
                    antecedent.span

    @staticmethod
    def compute_antecedent(mention, multigraph):
        weights = []
        for antecedent in multigraph.edges[mention]:
            if not multigraph.edges[mention][antecedent]["negative_relations"]:
                weights.append(
                    (multigraph.get_weight(mention, antecedent), antecedent))

        # get antecedent with highest positive weight, break ties by distance
        if len(weights) > 0 and sorted(weights)[-1][0] > 0:
            return sorted(weights)[-1][1]
