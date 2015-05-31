""" Extract coreference information from pairwise predictions."""

__author__ = 'smartschat'


def best_first(substructures, labels, scores, coref_labels=None):
    """ Extract coreference clusters from coreference predictions.

    In particular, go through a list of anaphor-antecedent pairs, where
    pairs with the same anaphor are consecutive. Then, for each anaphor, the
    best-scoring antecedent is selected (this is also called best-first
    clustering). Ties are broken by position in the list: earlier items are
    preferred.

    Args:
        pairs (list((Mention, Mention))): A list of (anaphor, antecedent)
            pairs. If two items in the list have the same anaphor, they
            should be consecutive.
        predictions (list(int)): A list of predictions. This list should
            have the same length as the list of pairs. Each entry describes
            the absence of an arc (if the entry is 0), or the label of an arc
            (if the entry is > 0).
        scores (list(float)): A list of prediction scores. This list should
            have the same length as the list of pairs. Each entry describes
            the score of an anaphor/antecedent decision.
        coref_labels (set(int)): A list of labels that indicate that mentions
            connected via an arc that has one of these labels are coreferent.

    Returns
        A tuple containing two dicts. The components are

            - **mention_entity_mapping** (*dict(Mention, int)*): A mapping of
              mentions to entity identifiers.
            - **antecedent_mapping** (*dict(Mention, Mention)*): A mapping of
              mentions to their antecedent (as determined by the
              ``coref_extractor``).
    """
    if not coref_labels:
        coref_labels = {"+"}

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

    return mention_entity_mapping, antecedent_mapping


def all_ante(substructures, labels, scores, coref_labels=None):
    mention_entity_mapping = {}
    antecedent_mapping = {}

    for substructure in substructures:
        for pair in substructure:
            anaphor, antecedent = pair

            if antecedent.is_dummy():
                continue

            antecedent_mapping[anaphor] = antecedent

            if antecedent not in mention_entity_mapping:
                mention_entity_mapping[antecedent] = \
                        antecedent.document.system_mentions.index(antecedent)

            mention_entity_mapping[anaphor] = \
                mention_entity_mapping[antecedent]

    return mention_entity_mapping, antecedent_mapping
