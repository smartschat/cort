""" Extract coreference information from pairwise predictions."""

__author__ = 'smartschat'


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
        labels (list(list(str))): A list of arc scores. This list should
            have the same length as the list of substructures, and each inner
            list should contain only one element (as in ``substructures``).
            Each entry describes the score of an arc.
        coref_labels (set(str)): A list of labels that indicate that mentions
            connected via an arc that has one of these labels are coreferent.

    Returns
        A tuple containing two dicts. The components are

            - **mention_entity_mapping** (*dict(Mention, int)*): A mapping of
              mentions to entity identifiers.
            - **antecedent_mapping** (*dict(Mention, Mention)*): A mapping of
              mentions to their antecedent.
    """

    anaphor = None
    best = None
    max_val = float('-inf')

    mention_entity_mapping = {}
    antecedent_mapping = {}

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
                antecedent_mapping[anaphor] = best
                if best not in mention_entity_mapping:
                    mention_entity_mapping[best] = \
                        best.document.system_mentions.index(best)

                mention_entity_mapping[anaphor] = \
                    mention_entity_mapping[best]

            best = None
            max_val = float('-inf')

        if score > max_val and label in coref_labels:
            max_val = score
            best = current_antecedent

        anaphor = current_anaphor

    if anaphor and best and not best.is_dummy():
        antecedent_mapping[anaphor] = best
        if best not in mention_entity_mapping:
            mention_entity_mapping[best] = \
                best.document.system_mentions.index(best)

        mention_entity_mapping[anaphor] = \
            mention_entity_mapping[best]

    return mention_entity_mapping, antecedent_mapping


def all_ante(substructures, labels, scores, coref_labels):
    """ Extract coreference clusters from coreference predictions via transitive
    closure.

    In particular, go through all (anaphor, antecedent) pairs contained in
    ``substructures``, and obtain coreference clusters by transitive closure.

    Args:
        substructures (list(list((Mention, Mention)))): A list of substructures.
        labels (list(list(str))): Not used by this function.
        labels (list(list(str))): Not used by this function.
        coref_labels (set(str)): Not used by this function.

    Returns
        A tuple containing two dicts. The components are

            - **mention_entity_mapping** (*dict(Mention, int)*): A mapping of
              mentions to entity identifiers.
            - **antecedent_mapping** (*dict(Mention, Mention)*): A mapping of
              mentions to their antecedent.
    """
    mention_entity_mapping = {}
    antecedent_mapping = {}

    for substructure in substructures:
        for pair in substructure:
            anaphor, antecedent = pair

            # skip dummy antecedents
            if antecedent.is_dummy():
                continue

            antecedent_mapping[anaphor] = antecedent

            # antecedent is not in the mapping: we initialize a new coreference
            # chain
            if antecedent not in mention_entity_mapping:
                # chain id: index of antecedent in system mentions
                mention_entity_mapping[antecedent] = \
                        antecedent.document.system_mentions.index(antecedent)

            # assign id based on antecedent
            mention_entity_mapping[anaphor] = \
                mention_entity_mapping[antecedent]

    return mention_entity_mapping, antecedent_mapping
