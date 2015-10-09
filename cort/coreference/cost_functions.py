""" Cost functions used during learning of coreference predictors. """

__author__ = 'martscsn'


def cost_based_on_consistency(arc, label="+"):
    """ Assign cost to arcs based on consistency of decision and anaphoricity.

    An anaphor-antecedent decision is consistent if either
        (a) the mentions are coreferent, or
        (b) the antecedent is the dummy mention, and the anaphor does not have
            any preceding coreferent mention among all extracted mentions.

    Note that (b) also contains cases where the mention has an antecedent in the
    gold data, but we were unable to extract this antecedent due to errors in
    mention detection.

    If the anaphor-antecedent decision represented by ``arc``is consistent, it
    gets cost 0. If the the decision is not consistent, and the antecedent is
    the dummy mention, it gets cost 2. Otherwise, it gets cost 1.

    Args:
        arc ((Mention, Mention)): A pair of mentions.
        label (str): The label to predict for the arc. Defaults to '+'.

    Return:
        (int): The cost of predicting the arc.
    """
    ana, ante = arc

    consistent = ana.decision_is_consistent(ante)

    # false new
    if not consistent and ante.is_dummy():
        return 2
    # wrong link
    elif not consistent:
        return 1
    else:
        return 0


def null_cost(arc, label="+"):
    """ Dummy cost function which always returns 0 (corresponding to not using
    a cost function at all).

    Args:
        arc ((Mention, Mention)): A pair of mentions.
        label (str): The label to predict for the arc. Defaults to '+'

    Return:
        0
    """
    return 0