import sys


__author__ = 'smartschat'


def get_label(tree):
    if sys.version_info[0] == 2:
        return tree.node
    else:
        return tree.label()