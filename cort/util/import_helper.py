import numpy
import pyximport
pyximport.install(setup_args={"include_dirs": numpy.get_include()})

import importlib
import inspect


__author__ = 'martscsn'


def import_from_path(name):
    splitted = name.split(".")
    package_name = ".".join(splitted[:-1])
    cls = splitted[-1]

    package = importlib.import_module(package_name)

    imported = getattr(package, cls)

    return imported


def get_features(filename):
    mention_features = []
    pairwise_features = []

    for line in open(filename).readlines():
        feature = import_from_path(line.strip())
        number_of_arguments = len(inspect.getargspec(feature)[0])

        if number_of_arguments == 1:
            mention_features.append(feature)
        elif number_of_arguments == 2:
            pairwise_features.append(feature)
        else:
            raise ValueError("Features must have one or two arguments, "
                             "feature " + line.strip() + " has " +
                             str(number_of_arguments) + " arguments.")

    return mention_features, pairwise_features
