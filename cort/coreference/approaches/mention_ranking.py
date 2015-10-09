""" Implements instance extraction and decoding for mention-ranking models.

This module implements two variants of the mention ranking model within a
framework that expresses coreference resolution as predicting latent structures,
while performing learning using a latent structured perceptron with
cost-augmented inference.

Hence, both variants are expressed as predicting a latent graph. In particular,
let m_1, ..., m_n be all mentions in a document. Let m_0 be a dummy mention for
anaphoricity determination. For an anaphor m_j, we predict the graph with nodes
m_0, ..., m_j and with the arc (m_j, m_i) if m_i is selected as the antecedent
of m_j. Such a graph is called a *substructure* (compared to the graph which
summarizes the decisions for the whole document).

The two variants implemented here are

    - latent antecedents for training (Chang et al., 2012): when learning
      weights, compare the prediction (m_j,m_i) with the best-scoring
      prediction (m_j,m_k) s.t. m_j and m_k are coreferent
    - closest antecedents for training (Denis and Baldridge, 2008): when
      learning weights, compare the prediction (m_j,m_i) with (m_j,m_k), where
      m_k is the closest antecedent of m_j.

To implement these variants, this module contains a function that defines the
search space for the graphs, and two decoders: one decoder computes the
best-scoring antecedent prediction and the best-scoring coreferent antecedent,
while the other computes the best-scoring antecedent prediction and the closest
antecedent.

References:

    - Pascal Denis and Jason Baldridge. 2008. Specialized models and ranking
      for coreference resolution. In *Proceedings of the 2008 Conference on
      Empirical Methods in Natural Language Processing*, Waikiki, Honolulu,
      Hawaii, 25-27 October 2008, pages 660-669.
      http://www.aclweb.org/anthology/D08-1069
    - Kai-Wei Chang, Rajhans Samdani, Alla Rozovskaya, Mark Sammons, and
      Dan Roth. 2012. Illinois-Coref: The UI system in the CoNLL-2012 shared
      task. In *Proceedings of the Shared Task of the 16th Conference on
      Computational Natural Language Learning*, Jeju Island, Korea, 12-14 July
      2012, pages 113-117.
      http://www.aclweb.org/anthology/W12-4513
"""


from cort.coreference import perceptrons


__author__ = 'martscsn'


def extract_substructures(doc):
    """ Extract the search space for the mention ranking model,

    The mention ranking model consists in computing the optimal antecedent
    for an anaphor, which corresponds to predicting an edge in graph. This
    functions extracts the search space for each such substructure (one
    substructure corresponds to one antecedent decision for an anaphor).

    The search space is represented as a nested list of mention pairs. The
    mention pairs are candidate arcs in the graph. The ith list contains all
    potential (mention, antecedent) pairs for the ith mention in the
    document. The antecedents are ordered by distance. For example,
    the third list contains the pairs (m_3, m_2), (m_3, m_1), (m_3, m_0),
    where m_j is the jth mention in the document.

    Args:
        doc (CoNLLDocument): The document to extract substructures from.

    Returns:
        (list(list((Mention, Mention)))): The nested list of mention pairs
        describing the search space for the substructures.
    """
    substructures = []

    # iterate over mentions
    for i, ana in enumerate(doc.system_mentions):
        for_anaphor_arcs = []

        # iterate in reversed order over candidate antecedents
        for ante in sorted(doc.system_mentions[:i], reverse=True):
            for_anaphor_arcs.append((ana, ante))

        substructures.append(for_anaphor_arcs)

    return substructures


class RankingPerceptron(perceptrons.Perceptron):
    """ A perceptron for mention ranking with latent antecedents. """
    def argmax(self, substructure, arc_information):
        """ Decoder for mention ranking with latent antecedents.

        Compute highest-scoring antecedent and highest-scoring antecedent
        consistent with gold coreference information for one anaphor.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. For mention
                ranking, this list consists of all potential anaphor-antecedent
                pairs for one fixed anaphor in descending order, such as
                    (m_3, m_2), (m_2, m_1), (m_2, m_0)
            arc_information (dict((Mention, Mention), (array, array, bool)): A
                mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features (represented as
                an int array via feature hashing), the costs for the arc (for
                each label, order as in self.get_labels()), and whether
                predicting the arc to be coreferent is consistent with the gold
                annotation).

        Returns:
            A 6-tuple describing the highest-scoring anaphor-antecedent
            decision, and the highest-scoring anaphor-antecedent decision
            consistent with the gold annotation. The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the
                  highest-scoring antecedent decision (the list contains only
                  one arc),
                - **best_labels** (*list(str)*): empty, the ranking approach
                  does not employ any labels,
                - **best_scores** (*list(float)*): the score of the
                  highest-scoring antecedent decision,
                - **best_cons_arcs** (*list((Mention, Mention))*): the
                  highest-scoring antecedent decision consistent with the gold
                  annotation (the list contains only one arc),
                - **best_cons_labels** (*list(str)*): empty, the ranking
                  approach does not employ any labels
                - **best_cons_scores** (*list(float)*): the score of the
                  highest-scoring antecedent decision consistent with the
                  gold information
                - **is_consistent** (*bool*): whether the highest-scoring
                  antecedent decision is consistent with the gold information.
        """
        best, max_val, best_cons, max_cons, best_is_consistent = \
            self.find_best_arcs(substructure, arc_information)

        return (
            [best],
            [],
            [max_val],
            [best_cons],
            [],
            [max_cons],
            best_is_consistent
        )


class RankingPerceptronClosest(perceptrons.Perceptron):
    """ A perceptron for mention ranking with closest antecedents for training.
    """
    def argmax(self, substructure, arc_information):
        """ Decoder for mention ranking with closest antecedents for training.

        Compute highest-scoring antecedent and closest gold antecedent for one
        anaphor.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. For mention
                ranking, this list consists of all potential anaphor-antecedent
                pairs for one fixed anaphor in descending order, such as
                    (m_3, m_2), (m_3, m_1), (m_3, m_0)
            arc_information (dict((Mention, Mention), (array, array, bool)): A
                mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features (represented as
                an int array via feature hashing), the costs for the arc (for
                each label, order as in self.get_labels()), and whether
                predicting the arc to be coreferent is consistent with the gold
                annotation).

        Returns:
            A 6-tuple describing the highest-scoring anaphor-antecedent
            decision, and the anaphor-antecedent pair with the closest gold
            antecedent. The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the
                  highest-scoring antecedent decision (the list contains only
                  one arc),
                - **best_labels** (*list(str)*): empty, the ranking approach
                  does not employ any labels,
                - **best_scores** (*list(float)*): the score of the
                  highest-scoring antecedent decision,
                - **best_cons_arcs** (*list((Mention, Mention))*): the
                  anaphor-antecedent pair with the closest gold antecedent (the
                  list contains only one arc),
                - **best_cons_labels** (*list(str)*): empty, the ranking
                  approach does not employ any labels
                - **best_cons_scores** (*list(float)*): the score of the
                  anaphor-antecedent pair with the closest gold antecedent
                - **is_consistent** (*bool*): whether the highest-scoring
                  antecedent decision is consistent with the gold information.
        """
        max_val = float("-inf")
        best = None

        max_cons = float("-inf")
        best_cons = None

        best_is_consistent = False

        for arc in substructure:
            score = self.score_arc(arc, arc_information)
            consistent = arc_information[arc][2]

            if score > max_val:
                best = arc
                max_val = score
                best_is_consistent = consistent

            # take closest
            if not best_cons and consistent:
                best_cons = arc
                max_cons = score

        return (
            [best],
            [],
            [max_val],
            [best_cons],
            [],
            [max_cons],
            best_is_consistent
        )