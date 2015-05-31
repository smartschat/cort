__author__ = 'martscsn'


def cost_based_on_consistency(arc):
    ana, ante = arc

    consistent = ana.decision_is_consistent(ante)

    if not consistent and ante.is_dummy():
        return 2
    elif not consistent:
        return 1
    else:
        return 0


def null_cost(arc):
    return 0