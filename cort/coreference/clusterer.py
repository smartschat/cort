""" Extract coreference information from pairwise predictions."""
from collections import defaultdict

from cort.util import union_find

__author__ = 'smartschat'


def closest_first(substructures, labels, scores, coref_labels):
    """ Extract coreference clusters from coreference predictions via closest-first
    clustering.

    In particular, go through a list of anaphor-antecedent pairs, where
    pairs with the same anaphor are consecutive. Then, for each anaphor, the
    first antecedent in the list is selected (this is also called closest-first
    clustering). For each anaphor, antecedents should be in the list in descending
    order with respect to the distance from the anaphor.

    Args:
        substructures (list(list((Mention, Mention)))): A list of substructures.
            For this clusterer, each substructure should contain only one
            (anaphor, antecedent) pair. If two substructures have the same
            anaphor, they should be consecutive.
        labels (list(list(str))): A list of arc labels. This list should
            have the same length as the list of substructures, and each inner
            list should contain only one element (as in ``substructures``).
            Each entry describes the label of an arc.
        scores (list(list(str))): Not used by this function.
        coref_labels (set(str)): A list of labels that indicate that mentions
            connected via an arc that has one of these labels are coreferent.

    Returns
        A tuple. The components are

            - **coref_sets** (*union_find.UnionFind*): An assignment of mentions
              to entities represented by a UnionFind data structure.
            - **antecedent_mapping** (*dict(Mention, Mention)*): A mapping of
              mentions to their antecedent.
    """

    anaphor = None
    best = None

    antecedent_mapping = defaultdict(list)

    union = union_find.UnionFind()

    for substructure, substructure_label in zip(
            substructures, labels):
        # each substructure consists of one pair
        pair = substructure[0]
        label = substructure_label[0]
        current_anaphor, current_antecedent = pair
        if current_anaphor != anaphor:
            # change in anaphor: set coreference information based on
            # best-scoring antecedent
            if anaphor and best and not best.is_dummy():
                antecedent_mapping[anaphor].append(best)
                union.union(anaphor, best)

            best = None

        if label in coref_labels and best is None:
            best = current_antecedent

        anaphor = current_anaphor

    if anaphor and best and not best.is_dummy():
        antecedent_mapping[anaphor].append(best)
        union.union(anaphor, best)

    return union, antecedent_mapping


def aggressive_merge(substructures, labels, scores, coref_labels):
    """ Extract coreference clusters from coreference predictions via aggressive-merge
    clustering.

    In particular, go through a list of anaphor-antecedent pairs. For each anaphor,
    all antecedents for which the pair is labeled as coreferent is selected.

    Args:
        substructures (list(list((Mention, Mention)))): A list of substructures.
            For this clusterer, each substructure should contain only one
            (anaphor, antecedent) pair.
        labels (list(list(str))): A list of arc labels. This list should
            have the same length as the list of substructures, and each inner
            list should contain only one element (as in ``substructures``).
            Each entry describes the label of an arc.
        scores (list(list(str))): Not used by this function.
        coref_labels (set(str)): A list of labels that indicate that mentions
            connected via an arc that has one of these labels are coreferent.

    Returns
        A tuple. The components are

            - **coref_sets** (*union_find.UnionFind*): An assignment of mentions
              to entities represented by a UnionFind data structure.
            - **antecedent_mapping** (*dict(Mention, Mention)*): A mapping of
              mentions to their antecedent.
    """
    antecedent_mapping = defaultdict(list)

    union = union_find.UnionFind()

    for substructure, substructure_label in zip(substructures, labels):
        # each substructure consists of one pair
        pair = substructure[0]
        label = substructure_label[0]
        current_anaphor, current_antecedent = pair

        if label in coref_labels and not current_antecedent.is_dummy():
            union.union(current_anaphor, current_antecedent)
            antecedent_mapping[current_anaphor].append(current_antecedent)

    return union, antecedent_mapping


def best_first(substructures, labels, scores, coref_labels):
    """ Extract coreference clusters from coreference predictions via best-first
    clustering.

    In particular, go through a list of anaphor-antecedent pairs, where
    pairs with the same anaphor are consecutive. Then, for each anaphor, the
    best-scoring antecedent is selected (this is also called best-first
    clustering). Ties are broken by position in the list: earlier items are
    preferred.

    Args:
        substructures (list(list((Mention, Mention)))): A list of substructures.
            For this clusterer, each substructure should contain only one
            (anaphor, antecedent) pair. If two substructures have the same
            anaphor, they should be consecutive.
        labels (list(list(str))): A list of arc labels. This list should
            have the same length as the list of substructures, and each inner
            list should contain only one element (as in ``substructures``).
            Each entry describes the label of an arc.
        scores (list(list(str))): A list of arc scores. This list should
            have the same length as the list of substructures, and each inner
            list should contain only one element (as in ``substructures``).
            Each entry describes the score of an arc.
        coref_labels (set(str)): A list of labels that indicate that mentions
            connected via an arc that has one of these labels are coreferent.

    Returns
        A tuple. The components are

            - **coref_sets** (*union_find.UnionFind*): An assignment of mentions
              to entities represented by a UnionFind data structure.
            - **antecedent_mapping** (*dict(Mention, Mention)*): A mapping of
              mentions to their antecedent.
    """

    anaphor = None
    best = None
    max_val = float('-inf')

    antecedent_mapping = defaultdict(list)

    union = union_find.UnionFind()

    for substructure, substructure_label, substructure_score in zip(
            substructures, labels, scores):
        # each substructure consists of one pair
        pair = substructure[0]
        label = substructure_label[0]
        score = substructure_score[0]
        current_anaphor, current_antecedent = pair
        if current_anaphor != anaphor:
            # change in anaphor: set coreference information based on
            # best-scoring antecedent
            if anaphor and best and not best.is_dummy():
                antecedent_mapping[anaphor].append(best)
                union.union(anaphor, best)

            best = None
            max_val = float('-inf')

        if score > max_val and label in coref_labels:
            max_val = score
            best = current_antecedent

        anaphor = current_anaphor

    if anaphor and best and not best.is_dummy():
        antecedent_mapping[anaphor].append(best)
        union.union(anaphor, best)

    return union, antecedent_mapping


def all_ante(substructures, labels, scores, coref_labels):
    """ Extract coreference clusters from coreference predictions via transitive
    closure.

    In particular, go through all (anaphor, antecedent) pairs contained in
    ``substructures``, and obtain coreference clusters by transitive closure.

    Args:
        substructures (list(list((Mention, Mention)))): A list of substructures.
        labels (list(list(str))): Not used by this function.
        scores (list(list(str))): Not used by this function.
        coref_labels (set(str)): Not used by this function.

    Returns
        A tuple. The components are

            - **coref_sets** (*union_find.UnionFind*): An assignment of mentions
              to entities represented by a UnionFind data structure.
            - **antecedent_mapping** (*dict(Mention, Mention)*): A mapping of
              mentions to their antecedent.
    """
    antecedent_mapping = defaultdict(list)

    union = union_find.UnionFind()

    for substructure in substructures:
        for pair in substructure:
            anaphor, antecedent = pair

            # skip dummy antecedents
            if antecedent.is_dummy():
                continue

            antecedent_mapping[anaphor].append(antecedent)
            union.union(anaphor, antecedent)

    return union, antecedent_mapping
