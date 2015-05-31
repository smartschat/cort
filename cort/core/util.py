""" Utility functions. """

__author__ = 'smartschat'


def clean_via_pos(tokens, pos):
    """ Clean a list of tokens according to their part-of-speech tags.

    In particular, retain only tokens which do not have the part-of-speech tag
    DT (determiner) or POS (possessive 's').

    Args:
        tokens (list(str)): A list of tokens.
        pos (list(str)): A list of corresponding part-of-speech tags.

    Returns:
        list(str): The list of tokens which do not have part-of-speech tag
        DT or POS.
    """
    return [token for token, pos in zip(tokens, pos)
            if pos not in ["DT", "POS"]]
