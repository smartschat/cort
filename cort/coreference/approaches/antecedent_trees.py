""" Implements instance extraction and decoding for antecedent trees.

This module implements antecedent trees (Fernandes et al., 2014) within a
framework that expresses coreference resolution as predicting latent structures,
while performing learning using a latent structured perceptron with
cost-augmented inference.

Hence, antecedent trees are expressed as as predicting a latent graph.
In particular, let m_1, ..., m_n be all mentions in a document. Let m_0 be a
dummy mention for anaphoricity determination. We predict
the graph with nodes m_0, ..., m_n and with arcs (m_j, m_i) which correspond to
antecedent decisions. In particular, for each j there exists exactly one i < j
such that (m_j, m_i) is in the graph. Such a graph is called aa *substructure*
(for antecedent trees, substructures and structures coincide).

To implement antecedent trees, this module contains a function that defines the
search space for the graphs, and a decoder that computes the best-scoring tree
of antecedent decisions, and the best-scoring tree of antecedent decisions
consistent with the gold annotation (i.e. only having pairs of coreferent
mentions as arcs).

Reference:

    - Eraldo Fernandes, Cicero dos Santos, and Ruy Milidiu. 2014. Latent trees
      for coreference resolution. *Computational Linguistics*, 40(4):801-835.
      http://www.aclweb.org/anthology/J14-4004
"""

from __future__ import division


from cort.coreference import perceptrons


__author__ = 'martscsn'


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


class AntecedentTreePerceptron(perceptrons.Perceptron):
    """ A perceptron for antecedent trees. """
    def argmax(self, substructure, arc_information):
        """ Decoder for antecedent trees.

        Compute highest-scoring antecedent tree and highest-scoring antecedent
        tree consistent with the gold annotation.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. For mention
                ranking, this list contains all potential anaphor-antecedent
                pairs in the following order:
                (m_1, m_0), (m_2, m_1), (m_2, m_0), (m_3, m_2), ...
            arc_information (dict((Mention, Mention), (array, array, bool)): A
                mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features (represented as
                an int array via feature hashing), the costs for the arc (for
                each label, order as in self.get_labels()), and whether
                predicting the arc to be coreferent is consistent with the gold
                annotation).

        Returns:
            A 6-tuple describing the highest-scoring antecedent tree, and the
            highest-scoring antecedent tree consistent with the gold
            annotation. The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the highest-scoring antecedent tree,
                - **best_labels** (*list(str)*): empty, the antecedent tree
                  approach does not employ any labels,
                - **best_scores** (*list(float)*): the scores of the
                  arcs in the highest-scoring antecedent tree,
                - **best_cons_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the highest-scoring antecedent tree consistent
                  with the gold annotation.
                - **best_cons_labels** (*list(str)*): empty, the antecedent
                  tree approach does not employ any labels
                - **best_cons_scores** (*list(float)*): the scores of the
                  arcs in the highest-scoring antecedent tree consistent with
                  the gold annotation,
                - **is_consistent** (*bool*): whether the highest-scoring
                  antecedent tree is consistent with the gold annotation.
        """
        if not substructure:
            return [], [], [], [], [], [], True

        number_mentions = len(substructure[0][0].document.system_mentions)

        arcs = []
        arcs_scores = []
        coref_arcs = []
        coref_arcs_scores = []

        is_consistent = True

        for ana_index in range(1, number_mentions):

            first_arc = ana_index*(ana_index-1)//2
            last_arc = first_arc + ana_index

            best, max_val, best_cons, max_cons, best_is_consistent = \
                self.find_best_arcs(substructure[first_arc:last_arc],
                                    arc_information)

            arcs.append(best)
            arcs_scores.append(max_val)
            coref_arcs.append(best_cons)
            coref_arcs_scores.append(max_cons)

            is_consistent &= best_is_consistent

        return (
            arcs,
            [],
            arcs_scores,
            coref_arcs,
            [],
            coref_arcs_scores,
            is_consistent
        )
