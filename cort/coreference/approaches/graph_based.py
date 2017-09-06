""" Implements instance extraction and decoding for antecedent graphs.

Antecedent graphs differ from antecedent trees by dropping the requirement that
each mention only has one antecedent.

This module implements such models within a framework that expresses coreference
resolution as predicting latent structures, while performing learning using a
latent structured perceptron with cost-augmented inference.

Let m_1, ..., m_n be all mentions in a document. Let m_0 be a dummy mention for
anaphoricity determination. We predict a graph with nodes m_0, ..., m_n and
with arcs A subset { (m_j, m_i) | j > i } (which correspond to antecedent decisions).
 Such a graph is called a *substructure* (for antecedent graphs, substructures and
 structures coincide).

To implement antecedent graphs, this module contains a function that defines the
search space for the graphs, and a decoder that computes the best-scoring graph
of antecedent decisions, and the best-scoring graph of antecedent decisions
consistent with the gold annotation (i.e. only having pairs of coreferent
mentions as arcs).

Furthermore, this model also contains a variant of antecedent graphs which
consider per-anaphor substructures (hence the resulting model can be considered a
variant of the mention-ranking model).
"""


from cort.coreference import perceptrons


__author__ = 'martscsn'


def extract_per_anaphor_substructures(doc):
    """ Extract the search space for the antecedent graph model that considers
    the induced graph for each anaphor as a substructure,

    The antecedent graph model consists in computing the antecedents for
    each anaphor. These decisions are represented as edges in a tree of
    anaphor-antecedent decisions. This functions extracts the search space for
    the each substructure (one substructure corresponds to the antecedent
    decisions for an anaphor).

    The search space is represented as a nested list of mention pairs. The
    mention pairs are candidate arcs in the graph. The ith list contains all
    potential (mention, antecedent) pairs for the ith mention in the
    document. The antecedents are ordered by distance. For example,
    the third list contains the pairs (m_3, m_2), (m_3, m_1), (m_3, m_0),

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


def extract_per_document_substructures(doc):
    """ Extract the search space for the document-wide antecedent graph model,

    The antecedent graph model consists in computing the antecedents for
    each anaphor. These decisions are represented as edges in a tree of
    anaphor-antecedent decisions. This functions extracts the search space for
    the graph.

    The search space is represented as a nested list of mention pairs. The
    mention pairs are candidate arcs in the graph. The nested list contains
    only one list, since antecedent trees have only one substructure for
    each document.

    The list contains all potential (anaphor, antecedent) pairs in the
    following order: (m_1, m_0), (m_2, m_1), (m_2, m_0), (m_3, m_2), ...,

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


class GraphPerceptronPerAnaphor(perceptrons.Perceptron):
    """ A perceptron for the per-anaphor variant of antecedent graphs. """
    def argmax(self, substructure, arc_information):
        """ Decoder for the per-anaphor variant of antecedent graphs.

        Compute highest-scoring antecedents and highest-scoring antecedents
        consistent with gold coreference information for one anaphor.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. This list
                consists of all potential anaphor-antecedent
                pairs for one fixed anaphor in descending order, such as
                    (m_3, m_2), (m_2, m_1), (m_2, m_0)
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
            A 9-tuple describing highest-scoring anaphor-antecedent
            decisions, and highest-scoring anaphor-antecedent decisions
            consistent with the gold annotation. The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the
                  highest-scoring antecedent decisions,
                - **best_labels** (*list(str)*): empty, the antecedent graph
                  approach does not employ any labels,
                - **best_scores** (*list(float)*): the score of the
                  highest-scoring antecedent decisions,
                - **best_additional_features** (*array(int)*): empty, this approach
                  does not employ any additional features.
                - **best_cons_arcs** (*list((Mention, Mention))*): the
                  highest-scoring antecedent decisions consistent with the gold
                  annotation,
                - **best_cons_labels** (*list(str)*): empty, this approach does not
                  employ any labels
                - **best_cons_scores** (*list(float)*): the score of the
                  highest-scoring antecedent decisions consistent with the
                  gold information.
                - **best_cons_additional_features** (*array(int)*): empty, this
                  approach does not employ any additional features.
                - **is_consistent** (*bool*): whether the selected
                  antecedent decision are consistent with the gold information.
        """
        arcs = []
        vals = []
        cons_arcs = []
        cons_vals = []
        consistent = True

        max_val = float("-inf")
        best = None

        max_cons = float("-inf")
        best_cons = None

        best_is_consistent = False

        for arc in substructure:
            arc_consistent = arc_information[arc][2]
            score = self.score_arc(arc, arc_information)

            if score > max_val:
                best = arc
                max_val = score
                best_is_consistent = arc_consistent

            if score > max_cons and arc_consistent:
                best_cons = arc
                max_cons = score

            if score > 0:
                arcs.append(arc)
                vals.append(score)

                if not arc_consistent:
                    consistent = False

            if arc_consistent:
                cons_arcs.append(arc)
                cons_vals.append(score)

        if not cons_arcs:
            cons_arcs = [best_cons]
            cons_vals = [max_cons]

        if not arcs:
            arcs = [best]
            vals = [max_val]

            consistent = best_is_consistent

        return (
            arcs,
            [],
            vals,
            [],
            cons_arcs,
            [],
            cons_vals,
            [],
            consistent
        )


class GraphPerceptronPerDocument(perceptrons.Perceptron):
    """ A perceptron for antecedent graphs. """
    def argmax(self, substructure, arc_information):
        """ Decoder for antecedent graphs.

        Compute the highest-scoring antecedent graph and the highest-scoring
        antecedent graph consistent with the gold annotation.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. For antecedent
                graphs, this list contains all potential anaphor-antecedent
                pairs in the following order:
                (m_1, m_0), (m_2, m_1), (m_2, m_0), (m_3, m_2), ...
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
            A 9-tuple describing the highest-scoring antecedent graph, and the
            highest-scoring antecedent graph consistent with the gold
            annotation. The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the highest-scoring antecedent graph,
                - **best_labels** (*list(str)*): empty, the antecedent graph
                  approach does not employ any labels,
                - **best_scores** (*list(float)*): the scores of the
                  arcs in the highest-scoring antecedent graph,
                - **best_additional_features** (*array(int)*): empty, the
                  antecedent graph approach does not employ any additional features.
                - **best_cons_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the highest-scoring antecedent graph consistent
                  with the gold annotation.
                - **best_cons_labels** (*list(str)*): empty, the antecedent
                  graph approach does not employ any labels
                - **best_cons_scores** (*list(float)*): the scores of the
                  arcs in the highest-scoring antecedent graph consistent with
                  the gold annotation,
                - **best_cons_additional_features** (*array(int)*): empty, the
                  antecedent graph approach does not employ any additional features.
                - **is_consistent** (*bool*): whether the highest-scoring
                  antecedent graph is consistent with the gold annotation.
        """
        number_mentions = len(substructure[0][0].document.system_mentions)

        arcs = []
        vals = []
        cons_arcs = []
        cons_vals = []
        consistent = True

        for ana_index in range(1, number_mentions):
            first_arc = ana_index*(ana_index-1)//2
            last_arc = first_arc + ana_index

            per_anaphor_arcs = []
            per_anaphor_arcs_scores = []
            per_anaphor_coref_arcs = []
            per_anaphor_coref_arcs_scores = []

            max_val = float("-inf")
            best = None

            max_cons = float("-inf")
            best_cons = None

            best_is_consistent = False

            for arc in substructure[first_arc:last_arc]:
                arc_consistent = arc_information[arc][2]
                score = self.score_arc(arc, arc_information)

                if score > max_val:
                    best = arc
                    max_val = score
                    best_is_consistent = arc_consistent

                if score > max_cons and arc_consistent:
                    best_cons = arc
                    max_cons = score

                if score > 0:
                    per_anaphor_arcs.append(arc)
                    per_anaphor_arcs_scores.append(score)

                    if not arc_consistent:
                        consistent = False

                if arc_consistent:
                    per_anaphor_coref_arcs.append(arc)
                    per_anaphor_coref_arcs_scores.append(score)

            if not per_anaphor_coref_arcs:
                per_anaphor_coref_arcs = [best_cons]
                per_anaphor_coref_arcs_scores = [max_cons]

            if not per_anaphor_arcs:
                per_anaphor_arcs = [best]
                per_anaphor_arcs_scores = [max_val]

                consistent &= best_is_consistent

            arcs.extend(per_anaphor_arcs)
            vals.extend(per_anaphor_arcs_scores)
            cons_arcs.extend(per_anaphor_coref_arcs)
            cons_vals.extend(per_anaphor_coref_arcs_scores)

        return (
            arcs,
            [],
            vals,
            [],
            cons_arcs,
            [],
            cons_vals,
            [],
            consistent
        )
