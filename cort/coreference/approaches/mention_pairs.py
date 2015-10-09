""" Implements instance extraction and decoding for a mention pair model.

This module implements a mention pair model (Soon et al. 2001) within a
framework that expresses coreference resolution as predicting latent structures,
while performing learning using a latent structured perceptron with
cost-augmented inference.

Hence, the mention pair model is expressed as  a latent graph. In particular,
let m_1, ..., m_n be all mentions in a document. We then consider the graph
with nodes m_1, ..., m_n and arcs (m_2, m_1), (m_3, m_2), (m_3, m_1),
(m_4, m_3), ...

Then each arc gets a label, either "+" (coreferent) or "-" (not coreferent).
As each arc is handled individually, each is called a *substructure*.

To implement the mention pair model, this module contains a decoder and
functions that define the search space for training and testing. The decoder
computes the highest-scoring label for an arc, and additionally returns the
true label. The function that defines the search space for testing extracts
arcs as decribed above. For training, arcs are filtered using a procedure
similar to the procedure decribed in Soon et al. (2001).

Reference:
    - Wee Meng Soon, Hwee Tou Ng, and Daniel Chung Yong Lim. 2001. A machine
      learning approach to coreference resolution of noun phrases.
      *Computational Linguistics*, 27(4):521-544.
      http://www.aclweb.org/anthology/J01-4004
"""


from cort.coreference import perceptrons


__author__ = 'martscsn'


def extract_training_substructures(doc):
    """ Extract the search space for training the mention pair model,

    The mention pair model consists in computing labels "+" (coreferent) and
    "-" (not coreferent) for mention pairs. We can view each such labeled pair
    as a labeled arc in a graph. As each pair is handled individually, each
    arc in this graph is a *substructure*.

    The search space is represented as a nested list of mention pairs. Each
    list in the nested list contains only one pair (the search for the optimal
    substructure only chooses a label).

    For training, the list of pairs is obtained via a heuristic similar to
    Soon et al. (2001). We employ the following algorithm: for every
    j >0, if the mention m_j is in some coreference chain, add all pairs
    (m_j, m_{j-1}), (m_j, m_{j-2}), ..., (m_j, m_i) to the list, where
    m_i is the first mention preceding m_j which is coreferent with m_j.

    Args:
        doc (CoNLLDocument): The document to extract substructures from.

    Returns:
        (list(list(Mention, Mention))): The nested list of mention pairs
        describing the search space for the substructures.
    """
    substructures = []

    # iterate over mentions
    for i, ana in enumerate(doc.system_mentions):
        if not ana.attributes["annotated_set_id"]:
            continue

        # iterate in reversed order over candidate antecedents
        for ante in sorted(doc.system_mentions[1:i], reverse=True):
            substructures.append([(ana, ante)])

            if ana.is_coreferent_with(ante):
                break

    return substructures


def extract_testing_substructures(doc):
    """ Extract the search space for predicting with the mention pair model,

    The mention ranking model consists in computing the optimal antecedent
    for an anaphor, which corresponds to predicting an edge in graph. This
    functions extracts the search space for each such substructure (one
    substructure corresponds to one antecedent decision for an anaphor).

    The search space is represented as a nested list of mention pairs. The
    mention pairs are candidate arcs in the graph. The ith list contains the
    ith mention pair, where we assume the following order:

    (m_2, m_1), (m_3, m_2), (m_3, m_1), (m_4, m_3), ...

    Args:
        doc (CoNLLDocument): The document to extract substructures from.

    Returns:
        (list(list(Mention, Mention))): The nested list of mention pairs
        describing the search space for the substructures.
    """
    substructures = []

    # iterate over mentions
    for i, ana in enumerate(doc.system_mentions):

        # iterate in reversed order over candidate antecedents
        for ante in sorted(doc.system_mentions[1:i], reverse=True):
            substructures.append([(ana, ante)])

    return substructures


class MentionPairsPerceptron(perceptrons.Perceptron):
    """ A perceptron for the mention pair model. """
    def argmax(self, substructure, arc_information):
        """ Decoder for the mention pair model.

        Compute highest-scoring label for a pair and the correct label for a
        pair.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. For the
                mention pair model, the list contains only one mention pair.
            arc_information (dict((Mention, Mention), (array, array, bool)): A
                mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features (represented as
                an int array via feature hashing), the costs for the arc (for
                each label, order as in self.get_labels()), and whether
                predicting the arc to be coreferent is consistent with the gold
                annotation).

        Returns:
            A 6-tuple describing the highest-scoring label for the pair, and
            the correct label for the pair. The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the pair under
                 consideration (the list contains only one arc),
                - **best_labels** (*list(str)*): the predicted label of the
                  pair. Is either '+' (coreferent) or '-' (not coreferent).
                - **best_scores** (*list(float)*): the score of the predicted
                  label.
                - **best_cons_arcs** (*list((Mention, Mention))*): the pair
                  under consideration (the list contains only one arc),
                - **best_cons_labels** (*list(str)*): the correct label for the
                  arc. Is either '+' (coreferent) or '-' (not coreferent).
                - **best_cons_scores** (*list(float)*): the score of the
                  correct label.
                - **is_consistent** (*bool*): whether the predicted label is the
                  same as the correct label.
        """
        arc = substructure[0]
        consistent = arc_information[arc][2]

        score_coref = self.score_arc(arc, arc_information, "+")
        score_non_coref = self.score_arc(arc, arc_information, "-")

        if score_coref >= score_non_coref:
            label = "+"
            score = score_coref
        else:
            label = "-"
            score = score_non_coref

        if consistent:
            coref_label = "+"
            coref_score = score_coref
        else:
            coref_label = "-"
            coref_score = score_non_coref

        return ([arc], [label], [score], [arc], [coref_label], [coref_score],
                label == coref_label)

    def get_labels(self):
        return ["+", "-"]
