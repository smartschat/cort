import pyximport
pyximport.install()

from cort.coreference import perceptrons
import numpy as np

__author__ = 'moosavi'


class SingletonPerceptron(perceptrons.Perceptron):
    """ A perceptron for the singleton model. """
    def argmax(self, substructure, arc_information):

        arc = substructure[0]
        consistent = arc_information[arc][2]

        score_coref = self.score_arc(arc, arc_information, "+")
        score_sing = self.score_arc(arc, arc_information, "-")

        if score_coref >= score_sing:
            label = "+"
            score = score_coref
        else:
            label = "-"
            score = score_sing

        if consistent:
            orig_label = "+"
            orig_score = score_coref
        else:
            orig_label = "-"
            orig_score = score_sing

        return ([arc], [label], [score], [arc], [orig_label], [orig_score],
                label == orig_label)

    def get_labels(self):
        return ["+", "-"]

    def set_singleton_scores (self, doc, arc_information):
        for mention in doc.system_mentions[1:]:
            mention.attributes["singletonScore"] = int(np.around(self.score_arc(mention, arc_information, "+")))


   
