""" Compute heads of mentions. """

import re

from cort.core import spans

__author__ = 'smartschat'


class HeadFinder:
    """Compute heads of mentions.

    This class provides functions to compute heads of mentions via modified
    version of the rules that can be found in Michael Collins' PhD thesis.
    The following changes were introduced:

        - handle NML as NP,
        - for coordinated phrases, take the coordination token as head,

    Furthermore, this class provides a function for adjusting heads for proper
    names to multi-token phrases via heuristics (see adjust_head_for_nam).
    """
    def __init__(self):
        self.__nonterminals = ["NP", "NML", "VP", "ADJP", "QP", "WHADVP", "S",
                             "ADVP", "WHNP", "SBAR", "SBARQ", "PP", "INTJ",
                             "SQ", "UCP", "X", "FRAG"]


        self.__nonterminal_rules = {
            "VP": (["TO", "VBD", "VBN", "MD", "VBZ", "VB", "VBG", "VBP", "VP",
                     "ADJP", "NN", "NNS", "NP"], False),
            "ADJP": (["NNS", "QP", "NN", "\$", "ADVP", "JJ", "VBN", "VBG", "ADJP",
              "JJR", "NP", "JJS", "DT", "FW", "RBR", "RBS", "SBAR", "RB"],
                     False),
            "QP": (["\$", "NNS", "NN", "IN", "JJ", "RB", "DT", "CD", "NCD",
              "QP", "JJR", "JJS"], False),
            "WHADVP": (["CC", "WRB"], True),
            "S": (["TO", "IN", "VP", "S", "SBAR", "ADJP", "UCP", "NP"], False),
            "SBAR": (["WHNP", "WHPP", "WHADVP", "WHADJP", "IN", "DT", "S", "SQ",
              "SINV", "SBAR", "FRAG"], False),
            "SBARQ": (["SQ", "S", "SINV", "SBARQ", "FRAG"], False),
            "SQ": (["VBZ", "VBD", "VBP", "VB", "MD", "VP", "SQ"], False),
            "ADVP": (["RB", "RBR", "RBS", "FW", "ADVP", "TO", "CD", "JJR", "JJ",
              "IN", "NP", "JJS", "NN"], True),
            "WHNP": (["WDT", "WP", "WP$", "WHADJP", "WHPP", "WHNP"], True),
            "PP": (["IN", "TO", "VBG", "VBN", "RP", "FW"], True),
            "X": (["S", "VP", "ADJP", "JJP", "NP", "SBAR", "PP", "X"], True),
            "FRAG": (["*"], True),
            "INTJ": (["*"], False),
            "UCP": (["*"], True),
        }

    def get_head(self, tree):
        """
        Compute the head of a mention, which is represented by its parse tree.

        Args:
            tree (nltk.ParentedTree): The parse tree of a mention.

        Returns:
            nltk.ParentedTree: The subtree of the input tree which corresponds
            to the head of the mention.
        """
        head = None

        label = tree.label()

        if len(tree) == 1:
            if tree.height() == 3:
                head = tree[0]
            elif tree.height() == 2:
                head = tree
        elif label in ["NP", "NML"]:
            head = self.__get_head_for_np(tree)
        elif label in self.__nonterminals:
            head = self.__get_head_for_nonterminal(tree)

        if head is None:
            head = self.get_head(tree[-1])

        return head

    def __get_head_for_np(self, tree):
        if self.__rule_cc(tree) is not None:
            return self.__rule_cc(tree)
        elif self.__collins_rule_nn(tree) is not None:
            return self.__collins_rule_nn(tree)
        elif self.__collins_rule_np(tree) is not None:
            return self.get_head(self.__collins_rule_np(tree))
        elif self.__collins_rule_nml(tree) is not None:
            return self.get_head(self.__collins_rule_nml(tree))
        elif self.__collins_rule_prn(tree) is not None:
            return self.__collins_rule_prn(tree)
        elif self.__collins_rule_cd(tree) is not None:
            return self.__collins_rule_cd(tree)
        elif self.__collins_rule_jj(tree) is not None:
            return self.__collins_rule_jj(tree)
        elif self.__collins_rule_last_word(tree) is not None:
            return self.__collins_rule_last_word(tree)

    def __get_head_for_nonterminal(self, tree):
        label = tree.label()
        values, traverse_reversed = self.__nonterminal_rules[label]
        if traverse_reversed:
            to_traverse = reversed(tree)
        else:
            to_traverse = tree
        for val in values:
            for child in to_traverse:
                label = child.label()
                if val == "*" or label == val:
                    if label in self.__nonterminals:
                        return self.get_head(child)
                    else:
                        return child

    def __rule_cc(self, tree):
        if tree.label() == "NP":
            for child in tree:
                if child.label() == "CC":
                    return child

    def __collins_rule_pos(self, tree):
        if tree.pos()[-1][1] == "POS":
            return tree[-1]

    def __collins_rule_nn(self, tree):
        for i in range(len(tree)-1, -1, -1):
            if re.match("NN|NNP|NNPS|JJR", tree[i].label()):
                return tree[i]
            elif tree[i].label() == "NX":
                return self.get_head(tree[i])

    def __collins_rule_np(self, tree):
        for child in tree:
            if child.label() == "NP":
                return child

    def __collins_rule_nml(self, tree):
        for child in tree:
            if child.label() == "NML":
                return child

    def __collins_rule_prn(self, tree):
        for child in tree:
            if child.label() == "PRN":
                return self.get_head(child[0])

    def __collins_rule_cd(self, tree):
        for i in range(len(tree)-1, -1, -1):
            if re.match("CD", tree[i].label()):
                return tree[i]

    def __collins_rule_jj(self, tree):
        for i in range(len(tree)-1, -1, -1):
            if re.match("JJ|JJS|RB", tree[i].label()):
                return tree[i]
            elif tree[i].label() == "QP":
                return self.get_head(tree[i])

    def __collins_rule_last_word(self, tree):
        current_tree = tree[-1]
        while current_tree.height() > 2:
            current_tree = current_tree[-1]

    @staticmethod
    def adjust_head_for_nam(tokens, pos, ner_type):
        """
        Adjust head for proper names via heuristics.

        Based on heuristics depending on the named entity type (person,
        organization, ...) and part-of-speech tags, adjust the head of a
        named entity mention to a meaningful extent useful for coreference
        resolution.

        For example, for the mention "Khan Younes in Southern Gaza Strip",
        this function will compute "Khan Younes" as the head.

        Args:
            tokens (list(str)): The tokens of the mention.
            pos (list(str)): The part-of-speech tags of the mention.
            ner_type (str): The named entity type of the mention. Should be
                one of PERSON, ORG, GPE, FAC, NORP, PRODUCT, EVENT, MONEY,
                WORK_OF_ART, LOC, LAW, LANGUAGE, DATE, TIME, ORDINAL,
                CARDINAL, QUANTITY, PERCENT or NONE.

        Returns:
            (Span, list(str)):
        """
        # TODO: get rid of this ugly hack
        if len(pos) == 0:
            return spans.Span(0, 0), "NOHEAD"

        stop_regex = re.compile("CC|,|\.|:|;|V.*|IN|W.*|ADVP|NN$")

        if re.match("ORG.*|GPE.*|FAC.*|NORP.*|PRODUCT|EVENT|MONEY|" +
                    "WORK_OF_ART|LOC.*|LAW|LANGUAGE", ner_type):
            start_regex = re.compile("NN(S)?|NNP(S)?")
            stop_regex = re.compile("V.*|IN|W.*|ADVP|,|-LRB-")
        elif ner_type == "PERSON":
            start_regex = re.compile("NN(S)?|NNP(S)?")
            stop_regex = re.compile("IN|CC|,|\.|:|;|V.*|W.*|-LRB-")
        elif re.match("DATE|TIME", ner_type):
            start_regex = re.compile("NN(S)?|NNP(S)?|CD")
        elif re.match("ORDINAL", ner_type):
            start_regex = re.compile("NN|JJ|RB")
        elif re.match("CARDINAL", ner_type):
            start_regex = re.compile("CD")
        elif re.match("QUANTITY|PERCENT", ner_type):
            start_regex = re.compile("CD|JJ|NN")
        elif ner_type == "NONE":
            start_regex = re.compile("NN(S)?|NNP(S)?|CD")
        else:
            raise Exception("Unknown named entity annotation: " + ner_type)

        head_start = -1

        position = 0

        for i in range(0, len(tokens)):
            position = i
            if head_start == -1 and start_regex.match(pos[i]):
                head_start = i
            elif head_start >= 0 and stop_regex.match(pos[i]):
                return spans.Span(head_start, i-1), tokens[head_start:i]

        if head_start == -1:
            head_start = 0

        if pos[position] == "POS" and position == len(pos) - 1:
            position -= 1

        return spans.Span(head_start, position), tokens[head_start:position+1]
