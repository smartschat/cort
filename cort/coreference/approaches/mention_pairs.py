""" Implements instance extraction and decoding for mention pair models.

This module implements mention pair models (Soon et al. 2001) within a
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
arcs as described above. For training, arcs are filtered using procedures
similar to the procedure described in Soon et al. (2001).

Reference:
    - Wee Meng Soon, Hwee Tou Ng, and Daniel Chung Yong Lim. 2001. A machine
      learning approach to coreference resolution of noun phrases.
      *Computational Linguistics*, 27(4):521-544.
      http://www.aclweb.org/anthology/J01-4004
"""


from cort.coreference import perceptrons


__author__ = 'martscsn'


def extract_training_substructures_soon(doc):
    """ Extract the search space for training the mention pair model,

    The mention pair model consists in computing labels "+" (coreferent) and
    "-" (not coreferent) for mention pairs. We can view each such labeled pair
    as a labeled arc in a graph. As each pair is handled individually, each
    arc in this graph is a *substructure*.

    The search space is represented as a nested list of mention pairs. Each
    list in the nested list contains only one pair (the search for the optimal
    substructure only chooses a label).

    For training, the list of pairs is obtained via a the heuristic of
    Soon et al. (2001). for every j > 0, if the mention m_j is anaphoric,
    add all pairs (m_j, m_{j-1}), (m_j, m_{j-2}), ..., (m_j, m_i) to the list,
    where m_i is the first mention preceding m_j which is coreferent with m_j.

    Args:
        doc (CoNLLDocument): The document to extract substructures from.

    Returns:
        (list(list(Mention, Mention))): The nested list of mention pairs
        describing the search space for the substructures.
    """
    substructures = []

    # iterate over mentions
    for i, ana in enumerate(doc.system_mentions):
        if ana.attributes["annotated_set_id"] is None:
            continue

        if ana.attributes["first_in_gold_entity"]:
            continue

        # iterate in reversed order over candidate antecedents
        for ante in sorted(doc.system_mentions[1:i], reverse=True):
            substructures.append([(ana, ante)])
            if ana.is_coreferent_with(ante):
                break

    return substructures


def extract_training_substructures_mod_soon(doc):
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
    j > 0, if the mention m_j is in some coreference chain, add all pairs
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
        if ana.attributes["annotated_set_id"] is None:
            continue

        # iterate in reversed order over candidate antecedents
        for ante in sorted(doc.system_mentions[:i], reverse=True):
            substructures.append([(ana, ante)])

            if ana.is_coreferent_with(ante):
                break

    return substructures


def extract_training_substructures_mod_soon_per_anaphor(doc):
    """ Extract the search space for training the mention pair model,

    The mention pair model consists in computing labels "+" (coreferent) and
    "-" (not coreferent) for mention pairs. We can view each such labeled pair
    as a labeled arc in a graph. This function considers the subgraph induced
    by each anaphor as a substructure.

    The search space is represented as a nested list of mention pairs. Each
    list in the nested list contains all pairs with the same anaphor.
    The search for the optimal substructure chooses labels for the pairs.

    For training, the list of pairs is obtained via a heuristic similar to
    Soon et al. (2001). We employ the following algorithm: for every
    j > 0, if the mention m_j is in some coreference chain, add all pairs
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
        if ana.attributes["annotated_set_id"] is None:
            continue

        to_add = []

        # iterate in reversed order over candidate antecedents
        for ante in sorted(doc.system_mentions[:i], reverse=True):
            to_add.append((ana, ante))

            if ana.is_coreferent_with(ante):
                break

        substructures.append(to_add)

    return substructures


def extract_training_substructures_mod_soon_per_document(doc):
    """ Extract the search space for training the mention pair model,

    The mention pair model consists in computing labels "+" (coreferent) and
    "-" (not coreferent) for mention pairs. We can view each such labeled pair
    as a labeled arc in a graph. This function considers the graph for the
    whole document as one substructure.

    The search space is represented as a nested list of mention pairs. The
    nested list contains only one list, since this variant of the mention pair
    model has only one substructure for each document. The search for the
    optimal substructure chooses labels for the pairs.

    For training, the list of pairs is obtained via a heuristic similar to
    Soon et al. (2001). We employ the following algorithm: for every
    j > 0, if the mention m_j is in some coreference chain, add all pairs
    (m_j, m_{j-1}), (m_j, m_{j-2}), ..., (m_j, m_i) to the list, where
    m_i is the first mention preceding m_j which is coreferent with m_j.

    Args:
        doc (CoNLLDocument): The document to extract substructures from.

    Returns:
        (list(list(Mention, Mention))): The nested list of mention pairs
        describing the search space for the substructures.
    """
    substructure = []

    # iterate over mentions
    for i, ana in enumerate(doc.system_mentions):
        if ana.attributes["annotated_set_id"] is None:
            continue

        # iterate in reversed order over candidate antecedents
        for ante in sorted(doc.system_mentions[:i], reverse=True):
            substructure.append((ana, ante))

            if ana.is_coreferent_with(ante):
                break

    return [substructure]


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
        for ante in sorted(doc.system_mentions[:i], reverse=True):
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
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.

        Returns:
            A 9-tuple describing the highest-scoring label for the pair, and
            the correct label for the pair. The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the pair under
                 consideration (the list contains only one arc),
                - **best_labels** (*list(str)*): the predicted label of the
                  pair. Is either '+' (coreferent) or '-' (not coreferent).
                - **best_scores** (*list(float)*): the score of the predicted
                  label.
                - **best_additional_features** (*array(int)*): empty, the
                  mention pair approach does not employ any additional features.
                - **best_cons_arcs** (*list((Mention, Mention))*): the pair
                  under consideration (the list contains only one arc),
                - **best_cons_labels** (*list(str)*): the correct label for the
                  arc. Is either '+' (coreferent) or '-' (not coreferent).
                - **best_cons_scores** (*list(float)*): the score of the
                  correct label.
                - **best_cons_additional_features** (*array(int)*): empty, the
                  mention pair approach does not employ any additional features.
                - **is_consistent** (*bool*): whether the predicted label is the
                  same as the correct label.
        """
        arcs, labels, scores = [], [], []
        coref_labels, coref_scores = [], []
        substructure_consistent = True

        for arc in substructure:
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

            arcs.append(arc)
            labels.append(label)
            scores.append(score)
            coref_labels.append(coref_label)
            coref_scores.append(coref_score)
            substructure_consistent &= label == coref_label

        return (arcs, labels, scores, [], arcs, coref_labels,
                coref_scores, [], substructure_consistent)

    def get_labels(self):
        """ Get the graph labels employed by the mention pair approach..

        The mention pair approach uses labels '+' (coreferent) and '-' (not
        coreferent).

        Returns:
            list(str): The list ['+', '-'] of graph labels.
        """
        return ["+", "-"]
