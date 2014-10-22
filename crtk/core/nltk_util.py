""" Utility functions for nltk 2/3 compatibility. """

import sys

from nltk import ParentedTree


__author__ = 'smartschat'


def parse_parented_tree(tree_string):
    """ Construct a tree from a constituent parse tree string.

    Args:
        tree_string (str): A constituent parse tree in bracket notation

    Returns:
        (nltk.ParentedTree): A parse tree corresponding to the parse tree
            string.
    """
    try:
        return ParentedTree(tree_string)
    except TypeError:
        return ParentedTree.fromstring(tree_string)


def get_label(tree):
    """ Get the label of the top node in the tree.

    For example, if the tree is (NP (DT the) (NN man)), returns "NP".

    Args:
        tree (nltk.ParentedTree): A parse tree.

    Returns:
        (str): the label of the top node in the tree.
    """
    try:
        return tree.node
    except NotImplementedError:
        return tree.label()


def get_lemma_name_of_first_synset(synsets):
    """ Get the first lemma name of the first synset in synsets.

    Args:
        synsets (list(nltk.corpus.reader.wordnet)): A list of synsets.

    Returns:
        (str): The first lemma of the first synset in synsets.
    """
    try:
        return synsets[0].lemma_names[0]
    except TypeError:
        return synsets[0].lemma_names()[0]