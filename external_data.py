import os
import pickle
from singletons import Singleton

__author__ = 'smartschat'

@Singleton
class GenderData:
    def __init__(self):
        self.word_to_gender = {}

        directory = os.path.dirname(os.path.realpath(__file__)) + "/resources/"

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
        # whole string
        if " ".join(attributes["tokens"]).lower() in self.word_to_gender:
            return self.word_to_gender[" ".join(attributes["tokens"]).lower()]
        # head
        elif " ".join(attributes["head"]).lower() in self.word_to_gender:
            return self.word_to_gender[" ".join(attributes["head"]).lower()]
        # head token by token
        elif self.look_up_token_by_token(attributes["head"]):
            return self.look_up_token_by_token(attributes["head"])

    def look_up_token_by_token(self, tokens):
        for token in tokens:
            if token[0].isupper() and token.lower() in self.word_to_gender:
                return self.word_to_gender[token.lower()]