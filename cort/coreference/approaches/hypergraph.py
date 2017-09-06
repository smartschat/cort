import array

from cort.core.mentions import Mention
from cort.coreference import perceptrons
from cort.coreference import data_structures
from cort.coreference.perceptrons import featurize

__author__ = 'martscsn'


class HypergraphPerceptron(perceptrons.Perceptron):
    """ A perceptron for entity based models. """
    def __init__(self, cost_scaling=1, cluster_features=None,
                 dynamic_features=None, mode="train"):
        super(HypergraphPerceptron, self).__init__(cost_scaling, cluster_features, dynamic_features)
        self.mode = mode

    def argmax(self, substructure, arc_information):
        cluster_feats = self.get_cluster_features()

        pred_additional_features = []
        cons_additional_features = []

        pred_arcs = []
        cons_arcs = []

        consistent = True

        dummy = Mention.dummy_from_document(substructure.mentions[0].document)

        for ana in substructure.mentions:
            if ana.is_dummy():
                continue

            clusters = []
            seen_clusters = set()

            for ante in reversed(substructure.mentions[:ana.attributes["id"]]):
                cluster = tuple(substructure.mentions_to_clusters_mapping[ante])
                if cluster in seen_clusters:
                    continue
                else:
                    seen_clusters.add(cluster)
                    clusters.append(cluster)

            scores = []
            scores_with_cost = []
            arcs_consistent = []
            arcs_cluster_features = []

            for cluster in clusters:
                # proxy: closest
                arc = (ana, cluster[0])

                ana, ante = arc

                score = self.score_arc(arc, arc_information, nocost=True)

                cluster_weight = 0

                if cluster != (dummy,) and cluster != ():
                    # generate cluster features...
                    cluster_features = featurize([ana], cluster, substructure,
                                                 cluster_feats)
                    cluster_nonnumeric_features, cluster_numeric_features, \
                    cluster_numeric_vals = cluster_features

                    # ...and compute weight
                    cluster_weight = self.get_weights(cluster_nonnumeric_features)
                    cluster_weight += self.get_weights(cluster_numeric_features,
                                                       vals=cluster_numeric_vals)

                    arcs_cluster_features.append(cluster_features[0] + cluster_features[1])
                else:
                    arcs_cluster_features.append([])

                score += cluster_weight

                scores.append(score)

                cost = self._get_cost(ana, cluster, arc_information)

                # assumes no labels
                scores_with_cost.append(score + cost)

                arc_consistent = False
                for m in cluster:
                    if arc_information[(ana, m)][2]:
                        arc_consistent = True
                        break

                arcs_consistent.append(arc_consistent)

            max_score = float("-inf")
            max_pair_index = None
            max_score_with_cost = float("-inf")
            max_with_cost_pair_index = None
            max_cons_score_with_cost = float("-inf")
            max_with_cost_cons_pair_index = None

            for k in range(len(clusters)):
                score = scores[k]

                if score > max_score:
                    max_score = score
                    max_pair_index = k

                score_with_cost = scores_with_cost[k]

                if score_with_cost > max_score_with_cost:
                    max_score_with_cost = score_with_cost
                    max_with_cost_pair_index = k

                if score_with_cost > max_cons_score_with_cost and arcs_consistent[k]:
                    max_cons_score_with_cost = score_with_cost
                    max_with_cost_cons_pair_index = k

            if self.mode == "train":
                substructure.add_link(ana, clusters[max_with_cost_cons_pair_index][0])
                pred_arcs.append((ana, clusters[max_with_cost_pair_index][0]))
                cons_arcs.append((ana, clusters[max_with_cost_cons_pair_index][0]))

                pred_additional_features.extend(arcs_cluster_features[max_with_cost_pair_index])
                cons_additional_features.extend(arcs_cluster_features[max_with_cost_cons_pair_index])

                if not arcs_consistent[max_with_cost_pair_index]:
                    consistent = False
            elif self.mode == "predict":
                substructure.add_link(ana, clusters[max_pair_index][0])
                pred_arcs.append((ana, clusters[max_pair_index][0]))
            else:
                raise Exception("Unknown mode for hypergraph perceptron: " + self.mode)

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

    def _convert_substructures(self, substructures):
        return [
            data_structures.Clustering(
                substructure[0][0].document.system_mentions)
            for substructure in substructures
            ]

    def _get_cost(self, ana, cluster, arc_information):
        raise NotImplementedError()


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
