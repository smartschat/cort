import array
from collections import defaultdict

from cort.coreference import perceptrons
from cort.coreference import data_structures


__author__ = 'martscsn'


class EntityPerceptron(perceptrons.Perceptron):
    """ A perceptron for entity based models. """
    def __init__(self, cost_scaling=1, cluster_features=None,
                 dynamic_features=None, mode="train"):
        super(EntityPerceptron, self).__init__(cost_scaling, cluster_features, dynamic_features)
        self.mode = mode

    def argmax(self, substructure, arc_information):

        cluster_feats = self.get_cluster_features()
        dynamic_feats = self.get_dynamic_features()

        pred_additional_features = []
        cons_additional_features = []

        pred_arcs = []
        cons_arcs = []

        consistent = True

        # rescore
        anaphor_to_scored_list, anaphor_to_scored_list_with_cost, anaphor_to_cons_scored_list_with_cost = \
            self._get_pair_scored_lists(substructure, arc_information)

        # compute
        while not substructure.every_mention_has_antecedent():
            if self.mode == "train":
                ana, ante = self._get_next(substructure,
                                           anaphor_to_scored_list,
                                           anaphor_to_scored_list_with_cost,
                                           anaphor_to_cons_scored_list_with_cost,
                                           cluster_feats,
                                           dynamic_feats,
                                           "train")

                cost_val, cost_pair, feats_cost = self._get_for_anaphor(ana,
                                                                        anaphor_to_scored_list_with_cost, substructure, cluster_feats, dynamic_feats)
                cons_val, cons_pair, feats_cons = self._get_for_anaphor(ana,
                                                                        anaphor_to_cons_scored_list_with_cost, substructure, cluster_feats, dynamic_feats)

                pred_arcs.append(cost_pair)
                cons_arcs.append(cons_pair)

                pred_additional_features.extend(feats_cost)
                cons_additional_features.extend(feats_cons)

                if not arc_information[cost_pair][2]:
                    consistent = False

                substructure.add_link(ana, ante)
            elif self.mode == "predict":
                ana, ante = self._get_next(substructure,
                                           anaphor_to_scored_list,
                                           anaphor_to_scored_list_with_cost,
                                           anaphor_to_cons_scored_list_with_cost,
                                           cluster_feats,
                                           dynamic_feats,
                                           "predict")
                substructure.add_link(ana, ante)
                pred_arcs.append((ana, ante))
            else:
                raise Exception("Unknown mode for entity perceptron: " + self.mode)

        return (
            pred_arcs,
            [],
            [0] * len(pred_arcs),
            array.array("I", pred_additional_features),
            cons_arcs,
            [],
            [0] * len(cons_arcs),
            array.array("I", cons_additional_features),
            consistent
        )

    def _get_pair_scored_lists(self, substructure, arc_information):
        anaphor_to_scored_list = defaultdict(list)
        anaphor_to_scored_list_with_cost = defaultdict(list)
        anaphor_to_cons_scored_list_with_cost = defaultdict(list)

        # score all pairs
        for j, ana in enumerate(substructure.mentions):
            for i, ante in enumerate(substructure.mentions[:j]):
                arc = (ana, ante)
                score = self.score_arc(arc, arc_information, nocost=True)

                anaphor_to_scored_list[ana].append((score, ante))

                anaphor_to_scored_list_with_cost[ana].append(
                    (score + self.cost_scaling * arc_information[arc][1][0],
                     ante)
                )

                if arc_information[arc][2]:
                    anaphor_to_cons_scored_list_with_cost[ana].append(
                        (score + self.cost_scaling * arc_information[arc][1][
                            0], ante)
                    )

        # map anaphors to top pairs
        for ana in substructure.mentions[1:]:
            anaphor_to_scored_list[ana] = sorted(
                anaphor_to_scored_list[ana], reverse=True)
            anaphor_to_scored_list_with_cost[ana] = sorted(
                anaphor_to_scored_list_with_cost[ana],reverse=True)
            anaphor_to_cons_scored_list_with_cost[ana] = sorted(
                anaphor_to_cons_scored_list_with_cost[ana], reverse=True)

        return anaphor_to_scored_list, anaphor_to_scored_list_with_cost, anaphor_to_cons_scored_list_with_cost

    def _get_next(self, substructure, anaphor_to_scored_list,
                  anaphor_to_scored_list_with_cost,
                  anaphor_to_cons_scored_list_with_cost, cluster_feats,
                  dynamic_feats, mode):
        raise NotImplementedError()

    def _get_for_anaphor(self,
                         ana,
                         list_to_consider,
                         substructure,
                         cluster_feats,
                         dynamic_feats):
        raise NotImplementedError()

    def _convert_substructures(self, substructures):
        return [
            data_structures.Clustering(
                substructure[0][0].document.system_mentions)
            for substructure in substructures
            ]


def extract_substructures(doc):
    """ Extract the search space for the antecedent tree model,

    The mention ranking model consists in computing the optimal antecedent for
    each anaphor. These decisions are represented as edges in a tree of
    anaphor-antecedent decisions. This functions extracts the search space for
    the tree.

    The search space is represented as a nested list of mention pairs. The
    mention pairs are candidate arcs in the graph. The nested list contains
    only one list, since antecedent trees have only one substructure for
    each document.

    The list contains all potential (anaphor, antecedent) pairs in the
    following order: (m_1, m_0), (m_2, m_1), (m_2, m_0), (m_3, m_2), ...,
    where m_j is the jth mention in the document.

    Args:
        doc (CoNLLDocument): The document to extract substructures from.

    Returns:
        (list(list(Mention, Mention))): The nested list of mention pairs
        describing the search space for the substructures.
    """
    substructure = []

    # iterate over mentions
    for i, ana in enumerate(doc.system_mentions):

        # iterate in reversed order over candidate antecedents
        for ante in sorted(doc.system_mentions[:i], reverse=True):
            substructure.append((ana, ante))

    return [substructure]
