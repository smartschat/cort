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

from queue import PriorityQueue
from collections import defaultdict

import logging

from cort.coreference import perceptrons
from cort.util.union_find import UnionFind

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
            A 7-tuple describing the highest-scoring antecedent tree, and the
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


class AntecedentTreePerceptronAgendaBasedKBest(perceptrons.Perceptron):
    def kbest(self, substructure, arc_information, k):
        """ Approximate k-best decoder for antecedent trees.

        Computes an approximation of the k highest-scoring antecedent trees
        by also considering suboptimal antecedent choices for mentions. Filters
        out trees that lead to the same coreference chains as trees that were
        already constructed and does so while generating the list.

        Should
        not be used during training.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. For mention
                ranking, this list contains all potential anaphor-antecedent
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
            k (int): Number of antecedent trees to include in the output.

        Returns:
            A list of 3-tuples describing the (approximately) k highest-scoring
            antecedent trees, ordered by decreasing total score. If less than k
            trees were created, pad the list with the lowest-scoring tree included and
            log a warning.

            Each tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the antecedent tree,
                - **best_labels** (*list(str)*): empty, the antecedent tree
                  approach does not employ any labels,
                - **best_scores** (*list(float)*): the scores of the
                  arcs in the antecedent tree.
        """
        if not substructure:
            return [], []

        number_mentions = len(substructure[0][0].document.system_mentions)

        # first compute mapping mention to (score, ante) pair
        mention_mapping = self._compute_mention_to_score_ante_mapping(
            number_mentions, substructure, arc_information
        )

        solutions = []
        coref_union_find = []

        # compute 1-best solution

        best_mapping, best_uf = self._compute_1_best_solution(mention_mapping)
        solutions.append(best_mapping)
        coref_union_find.append(best_uf)

        # sort all choices in priority queue
        my_queue = self._generate_queue(mention_mapping, best_mapping)

        # generate approximate k-best solutions
        while not my_queue.empty():
            if len(solutions) == k:
                break

            score, (mention, ante) = my_queue.get()

            novel = True

            for uf in coref_union_find:
                if uf[mention] == uf[ante]:
                    novel = False

            if ante.is_dummy():
                for sol in solutions:
                    if sol[mention][1].is_dummy():
                        novel = False

            if novel:
                new_mapping, new_uf = self._compute_new_mapping_and_uf(
                    solutions[0], mention, ante, score)

                solutions.append(new_mapping)
                coref_union_find.append(new_uf)

        # transform into correct output format
        output = []

        for map in solutions:
            arcs, arcs_scores = self._transform_from_mapping(map)
            output.append((arcs, [], arcs_scores))

        output_length = len(output)

        if output_length < k:
            for i in range(output_length, k):
                output.append(output[output_length - 1])

            logging.warning("Less than " + str(k) + " trees included in k-best list. "
                            "Padded list with lowest-scoring.")

        return output

    def _compute_mention_to_score_ante_mapping(
            self, number_mentions, substructure, arc_information):
        # first compute mapping mention to (score, ante) pair
        mention_mapping = defaultdict(list)

        for ana_index in range(1, number_mentions):
            first_arc = ana_index * (ana_index - 1) // 2
            last_arc = first_arc + ana_index

            mention = substructure[first_arc][0]

            for arc in substructure[first_arc:last_arc]:
                ante = arc[1]
                score = self.score_arc(arc, arc_information)

                mention_mapping[mention].append((score, ante))

            mention_mapping[mention] = sorted(mention_mapping[mention])

        return mention_mapping

    def _compute_1_best_solution(self, mention_mapping):
        mapping = {}
        union_find = UnionFind()

        for mention in sorted(mention_mapping.keys()):
            score, antecedent = mention_mapping[mention][-1]
            mapping[mention] = (score, antecedent)

            if not antecedent.is_dummy():
                union_find.union(mention, antecedent)
            else:
                union_find.union(mention, mention)

        return mapping, union_find

    def _generate_queue(self, mention_mapping, best_mapping):
        my_queue = PriorityQueue()

        for mention in sorted(mention_mapping.keys()):
            best_score = best_mapping[mention][0]
            for score, ante in mention_mapping[mention]:
                my_queue.put((best_score - score, (mention, ante)))

        return my_queue

    def _compute_new_mapping_and_uf(self, base_mapping, mention, ante, score):
        my_uf = UnionFind()

        new_mapping = {}
        for m in base_mapping:
            if m != mention:
                new_mapping[m] = base_mapping[m]
                old_ante = base_mapping[m][1]
                if not old_ante.is_dummy():
                    my_uf.union(m, base_mapping[m][1])
                else:
                    my_uf.union(m, m)
            else:
                best_score = base_mapping[m][0]
                new_mapping[m] = (best_score - score, ante)
                if not ante.is_dummy():
                    my_uf.union(m, ante)
                else:
                    my_uf.union(m, m)

        return new_mapping, my_uf

    def _transform_from_mapping(self, mapping):
        arcs = []
        arcs_scores = []

        for mention in sorted(mapping.keys()):
            score, ante = mapping[mention]
            arcs.append((mention, ante))
            arcs_scores.append(score)

        return arcs, arcs_scores


class AntecedentTreePerceptronOvergeneratingKBest(perceptrons.Perceptron):
    def kbest(self, substructure, arc_information, k):
        """ Approximate k-best decoder for antecedent trees.

        Computes an approximation of the k highest-scoring antecedent trees
        by also considering suboptimal antecedent choices for mentions. First
        generates a list of k-best trees of the requested size and then, for each
        coreference chain equivalence class, only retains the highest-scoring tree
        that leads to this equivalence class. Should not be used during training.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure. For mention
                ranking, this list contains all potential anaphor-antecedent
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
            k (int): Number of antecedent trees created (actual k-best list may contain
                fewer elements).

        Returns:
            A list of 3-tuples describing the highest-scoring trees leading to different
            coreference chains, ordered by decreasing total score.

            Each tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the arcs
                  constituting the antecedent tree,
                - **best_labels** (*list(str)*): empty, the antecedent tree
                  approach does not employ any labels,
                - **best_scores** (*list(float)*): the scores of the
                  arcs in the antecedent tree.
        """
        if not substructure:
            return [], []

        number_mentions = len(substructure[0][0].document.system_mentions)

        # first compute mapping mention to (score, ante) pair
        mention_mapping = self._compute_mention_to_score_ante_mapping(
            number_mentions, substructure, arc_information
        )

        solutions = []
        coref_union_find = []

        # compute 1-best solution
        best_mapping, best_uf = self._compute_1_best_solution(mention_mapping)
        solutions.append(best_mapping)
        coref_union_find.append(best_uf)

        # sort all choices in priority queue
        my_queue = self._generate_queue(mention_mapping, best_mapping)

        # generate approximate k-best solutions
        while True:
            if len(solutions) >= k:
                break

            score, pair = my_queue.get()

            mention, ante = pair

            new_solutions = []

            for sol in solutions:
                new_solutions.append(sol)

            for sol in solutions:
                if sol[mention][1] != ante:
                    new_mapping, _ = self._compute_new_mapping_and_uf(sol, mention, ante, score)
                    new_solutions.append(new_mapping)

            solutions = new_solutions

        solutions = solutions[:k]

        # transform into correct output format
        temp_output = []

        for map in solutions:
            arcs, arcs_scores = self._transform_from_mapping(map)
            temp_output.append((sum(arcs_scores), (arcs, [], arcs_scores)))

        sorted_descending_temp_output = sorted(temp_output, reverse=True)

        final_output = []
        coref_for_computing_whether_equivalence_class_already_seen = []

        for sol in sorted_descending_temp_output:
            _, tree_description = sol
            arcs, _, _ = tree_description

            coref_chain = self._compute_uf_from_arcs(arcs).get_representation_for_comparison()

            already_added = False

            for x in coref_for_computing_whether_equivalence_class_already_seen:
                if coref_chain == x:
                    already_added = True

            if not already_added:
                final_output.append(tree_description)
                coref_for_computing_whether_equivalence_class_already_seen.append(coref_chain)

        print(len(final_output))

        return final_output

    def _compute_mention_to_score_ante_mapping(
            self, number_mentions, substructure, arc_information):
        # first compute mapping mention to (score, ante) pair
        mention_mapping = defaultdict(list)

        for ana_index in range(1, number_mentions):
            first_arc = ana_index * (ana_index - 1) // 2
            last_arc = first_arc + ana_index

            mention = substructure[first_arc][0]

            for arc in substructure[first_arc:last_arc]:
                ante = arc[1]
                score = self.score_arc(arc, arc_information)

                mention_mapping[mention].append((score, ante))

            mention_mapping[mention] = sorted(mention_mapping[mention])

        return mention_mapping

    def _compute_1_best_solution(self, mention_mapping):
        mapping = {}
        union_find = UnionFind()

        for mention in sorted(mention_mapping.keys()):
            score, antecedent = mention_mapping[mention][-1]
            mapping[mention] = (score, antecedent)

            if not antecedent.is_dummy():
                union_find.union(mention, antecedent)
            else:
                union_find.union(mention, mention)

        return mapping, union_find

    def _generate_queue(self, mention_mapping, best_mapping):
        my_queue = PriorityQueue()

        for mention in sorted(mention_mapping.keys()):
            best_score = best_mapping[mention][0]
            for score, ante in mention_mapping[mention]:
                my_queue.put((best_score - score, (mention, ante)))

        return my_queue

    def _compute_new_mapping_and_uf(self, base_mapping, mention, ante, score):
        my_uf = UnionFind()

        new_mapping = {}
        for m in base_mapping:
            if m != mention:
                new_mapping[m] = base_mapping[m]
                old_ante = base_mapping[m][1]
                if not old_ante.is_dummy():
                    my_uf.union(m, base_mapping[m][1])
                else:
                    my_uf.union(m, m)
            else:
                best_score = base_mapping[m][0]
                new_mapping[m] = (best_score - score, ante)
                if not ante.is_dummy():
                    my_uf.union(m, ante)
                else:
                    my_uf.union(m, m)

        return new_mapping, my_uf

    def _compute_uf_from_arcs(self, arcs):
        my_uf = UnionFind()

        for arc in arcs:
            ana, ante = arc

            if not ante.is_dummy():
                my_uf.union(ana, ante)
            else:
                my_uf.union(ana, ana)

        return my_uf

    def _transform_from_mapping(self, mapping):
        arcs = []
        arcs_scores = []

        for mention in sorted(mapping.keys()):
            score, ante = mapping[mention]
            arcs.append((mention, ante))
            arcs_scores.append(score)

        return arcs, arcs_scores
