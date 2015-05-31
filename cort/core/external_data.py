""" Read in and access data from external resources such as gender lists."""

import os
import pickle


import cort
from cort.core import singletons
from cort.core import util


__author__ = 'smartschat'


@singletons.Singleton
class GenderData:
    """ Read in and access data from lists with gender information.

    Attributes:
        word_to_gender (dict(str, str)): A mapping from lower-case strings
            to one of four genders: 'MALE', 'FEMALE', 'NEUTRAL' and 'PLURAL'.
    """
    def __init__(self):
        """ Initialize the word-to-gender mapping from gender lists.
        """
        self.word_to_gender = {}

        directory = cort.__path__[0] + "/resources/"

        lists = [
            open(directory + "male.list"),
            open(directory + "female.list"),
            open(directory + "neutral.list"),
            open(directory + "plural.list")
        ]

        genders = ["MALE", "FEMALE", "NEUTRAL", "PLURAL"]

        for gender, gender_list in zip(genders, lists):
            for word in gender_list.readlines():
                self.word_to_gender[word.strip()] = gender

    def look_up(self, attributes):
        """ Look up the gender of a mention described by the input attributes.

        Args:
            attributes (dict(str,object)): A dict describing attributes of
                mentions. Must contain "tokens" and "head", which have lists
                of strings as values.

        Returns:
            (str): None or one of the four genders 'MALE', 'FEMALE',
            'NEUTRAL' or 'PLURAL'.
        """
        # whole string
        if " ".join(attributes["tokens"]).lower() in self.word_to_gender:
            return self.word_to_gender[" ".join(attributes["tokens"]).lower()]
        # head
        elif " ".join(attributes["head"]).lower() in self.word_to_gender:
            return self.word_to_gender[" ".join(attributes["head"]).lower()]
        # head token by token
        elif self.__look_up_token_by_token(attributes["head"]):
            return self.__look_up_token_by_token(attributes["head"])

    def __look_up_token_by_token(self, tokens):
        for token in tokens:
            if token[0].isupper() and token.lower() in self.word_to_gender:
                return self.word_to_gender[token.lower()]


@singletons.Singleton
class LexicalData:
    """ Read in and access data containing pairs of coreferent mention strings.

    Attributes:
        pairs (set((str, str))): A set of string pairs, which represent strings
            of potentially coreferent mentions.
    """
    def __init__(self):
        """ Initialize the set of pairs from
            package_root/resources/coreferent_pairs.obj.
        """
        directory = cort.__path__[0] + "/resources/"

        self.pairs = pickle.load(
            open(directory + "coreferent_pairs.obj", "rb"))

    def look_up(self, anaphor, antecedent):
        """ Look up strings of the mentions in the pair list.

        Args:
            anaphor (Mention): A mention.
            antecedent (Mention): Another mention, the candidate antecedent
                for anaphor.

        Returns:
            True if the pair of strings corresponding to anaphor of
            antecedent, stripped determiners and possessive s, can be found
            in the list of pairs.
        """
        # whole string
        anaphor_cleaned = " ".join(
            util.clean_via_pos(anaphor.attributes["tokens"],
                          anaphor.attributes["pos"]))
        antecedent_cleaned = " ".join(
            util.clean_via_pos(antecedent.attributes["tokens"],
                               antecedent.attributes["pos"]))

        return (
            (anaphor_cleaned, antecedent_cleaned) in self.pairs
            or (antecedent_cleaned, anaphor_cleaned) in self.pairs
        )


@singletons.Singleton
class SingletonMentions:
    """ Read in and access data strings of singleton mentions.

    Attributes:
        singletons (set(str)): A set of strings, which represent strings of
            of potential singleton mentions.
    """
    def __init__(self):
        """ Initialize the set of pairs from
            package_root/resources/singletons_not_cleaned.obj.
        """
        directory = cort.__path__[0] + "/resources/"

        self.singletons = pickle.load(
            open(directory + "singletons_not_cleaned.obj", "rb"))
