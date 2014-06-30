from __future__ import print_function, division
import re
import nltk_util
from spans import Span


__author__ = 'smartschat'


class HeadFinder:
    @staticmethod
    def get_head(tree):
        head = None

        label = nltk_util.get_label(tree)

        if len(tree) == 1:
            if tree.height() == 3:
                head = tree[0]
            elif tree.height() == 2:
                head = tree
        elif label == "NP":
            head = HeadFinder.get_head_np(tree)
        elif label == "NML":
            head = HeadFinder.get_head_np(tree)
        elif label == "VP":
            head = HeadFinder.get_head_vp(tree)
        elif label == "ADJP":
            head = HeadFinder.get_head_adjp(tree)
        elif label == "QP":
            head = HeadFinder.get_head_qp(tree)
        elif label == "WHADVP":
            head = HeadFinder.get_head_whatadvp(tree)
        elif label == "S":
            head = HeadFinder.get_head_s(tree)
        elif label == "ADVP":
            head = HeadFinder.get_head_advp(tree)
        elif label == "WHNP":
            head = HeadFinder.get_head_whnp(tree)
        elif label == "SBAR":
            head = HeadFinder.get_head_sbar(tree)
        elif label == "SBARQ":
            head = HeadFinder.get_head_sbarq(tree)
        elif label == "PP":
            head = HeadFinder.get_head_pp(tree)
        elif label == "INTJ":
            head = HeadFinder.get_head_intj(tree)
        elif label == "SQ":
            head = HeadFinder.get_head_sq(tree)
        elif label == "UCP":
            head = HeadFinder.get_head_ucp(tree)
        elif label == "X":
            head = HeadFinder.get_head_x(tree)
        elif label == "FRAG":
            head = HeadFinder.get_head_frag(tree)

        if head is None:
            head = HeadFinder.get_head(tree[-1])

        return head

    @staticmethod
    def get_head_np(tree):
        if HeadFinder.rule_CC(tree) is not None:
            return HeadFinder.rule_CC(tree)
        elif HeadFinder.collins_rule_NN(tree) is not None:
            return HeadFinder.collins_rule_NN(tree)
        elif HeadFinder.collins_rule_NP(tree) is not None:
            return HeadFinder.get_head(HeadFinder.collins_rule_NP(tree))
        elif HeadFinder.collins_rule_NML(tree) is not None:
            return HeadFinder.get_head(HeadFinder.collins_rule_NML(tree))
        elif HeadFinder.collins_rule_PRN(tree) is not None:
            return HeadFinder.collins_rule_PRN(tree)
        elif HeadFinder.collins_rule_CD(tree) is not None:
            return HeadFinder.collins_rule_CD(tree)
        elif HeadFinder.collins_rule_JJ(tree) is not None:
            return HeadFinder.collins_rule_JJ(tree)
        elif HeadFinder.collins_rule_last_word(tree) is not None:
            return HeadFinder.collins_rule_last_word(tree)

    @staticmethod
    def get_head_vp(tree):
        values = ["TO", "VBD", "VBN", "MD", "VBZ", "VB", "VBG", "VBP", "VP", "ADJP", "NN", "NNS", "NP"]
        for val in values:
            for i in range(0, len(tree)):
                if nltk_util.get_label(tree[i]) == val:
                    if re.match("VP|NP|ADJP", val):
                        return HeadFinder.get_head(tree[i])
                    else:
                        return tree[i]

    @staticmethod
    def get_head_adjp(tree):
        values = ["NNS", "QP", "NN", "\$", "ADVP", "JJ", "VBN", "VBG", "ADJP", "JJR", "NP", "JJS", "DT", "FW", "RBR",
                  "RBS", "SBAR", "RB"]

        for val in values:
            for i in range(0, len(tree)):
                if nltk_util.get_label(tree[i]) == val:
                    if re.match("QP|ADVP|ADJP|NP,SBAR", val):
                        return HeadFinder.get_head(tree[i])
                    else:
                        return tree[i]

    @staticmethod
    def get_head_qp(tree):
        values = ["\$", "NNS", "NN", "IN", "JJ", "RB", "DT", "CD", "NCD", "QP", "JJR", "JJS"]

        for val in values:
            for i in range(0, len(tree)):
                if nltk_util.get_label(tree[i]) == val:
                    if re.match("QP", val):
                        return HeadFinder.get_head(tree[i])
                    else:
                        return tree[i]

    @staticmethod
    def get_head_whatadvp(tree):
        values = ["CC", "WRB"]

        for val in values:
            for i in range(len(tree)-1, -1, -1):
                if nltk_util.get_label(tree[i]) == val:
                    return tree[i]

    @staticmethod
    def get_head_s(tree):
        values = ["TO", "IN", "VP", "S", "SBAR", "ADJP", "UCP", "NP"]

        for val in values:
            for i in range(0, len(tree)):
                if nltk_util.get_label(tree[i]) == val:
                    if re.match("VP|S|SBAR|ADJP|UCP|NP", val):
                        return HeadFinder.get_head(tree[i])
                    else:
                        return tree[i]

    @staticmethod
    def get_head_sbar(tree):
        values = ["WHNP", "WHPP", "WHADVP", "WHADJP", "IN", "DT", "S", "SQ", "SINV", "SBAR", "FRAG"]

        for val in values:
            for i in range(0, len(tree)):
                if nltk_util.get_label(tree[i]) == val:
                    if re.match("WHNP|WHPP|WHADVP|WHADJP|S|SQ|SINV|SBAR|FRAG", val):
                        return HeadFinder.get_head(tree[i])
                    else:
                        return tree[i]

    @staticmethod
    def get_head_sbarq(tree):
        values = ["SQ", "S", "SINV", "SBARQ", "FRAG"]

        for val in values:
            for i in range(0, len(tree)):
                if nltk_util.get_label(tree[i]) == val:
                    if re.match("SQ|S|SINV|SBARQ|FRAG", val):
                        return HeadFinder.get_head(tree[i])
                    else:
                        return tree[i]

    @staticmethod
    def get_head_sq(tree):
        values = ["VBZ", "VBD", "VBP", "VB", "MD", "VP", "SQ"]

        for val in values:
            for i in range(0, len(tree)):
                if nltk_util.get_label(tree[i]) == val:
                    if re.match("VP|SQ", val):
                        return HeadFinder.get_head(tree[i])
                    else:
                        return tree[i]

    @staticmethod
    def get_head_advp(tree):
        values = ["RB", "RBR", "RBS", "FW", "ADVP", "TO", "CD", "JJR", "JJ", "IN", "NP", "JJS", "NN"]

        for val in values:
            for i in range(len(tree)-1, -1, -1):
                if nltk_util.get_label(tree[i]) == val:
                    if re.match("NP|ADVP", val):
                        return HeadFinder.get_head(tree[i])
                    else:
                        return tree[i]

    @staticmethod
    def get_head_whnp(tree):
        values = ["WDT", "WP", "WP$", "WHADJP", "WHPP", "WHNP"]

        for val in values:
            for i in range(len(tree)-1, -1, -1):
                if nltk_util.get_label(tree[i]) == val:
                    if re.match("WHPP|WHNP", val):
                        return HeadFinder.get_head(tree[i])
                    else:
                        return tree[i]

    @staticmethod
    def get_head_pp(tree):
        values = ["IN", "TO", "VBG", "VBN", "RP", "FW"]

        for val in values:
            for i in range(len(tree)-1, -1, -1):
                if nltk_util.get_label(tree[i]) == val:
                    return tree[i]

    @staticmethod
    def get_head_frag(tree):
        return HeadFinder.get_head(tree[-1])

    @staticmethod
    def get_head_intj(tree):
        return HeadFinder.get_head(tree[0])

    @staticmethod
    def get_head_ucp(tree):
        return HeadFinder.get_head(tree[-1])

    @staticmethod
    def get_head_x(tree):
        values = ["S", "VP", "ADJP", "JJP", "NP", "SBAR", "PP", "X"]

        for val in values:
            for i in range(len(tree)-1, -1, -1):
                if nltk_util.get_label(tree[i]) == val:
                    if re.match("S|VP|ADJP|NP|SBAR|PP|X", val):
                        return HeadFinder.get_head(tree[i])
                    else:
                        return tree[i]

    @staticmethod
    def collins_rule_POS(tree):
        if tree.pos()[-1][1] == "POS":
            return tree[-1]

    @staticmethod
    def collins_rule_NN(tree):
        for i in range(len(tree)-1, -1, -1):
            if re.match("NN|NNP|NNPS|JJR", nltk_util.get_label(tree[i])):
                return tree[i]
            elif nltk_util.get_label(tree[i]) == "NX":
                return HeadFinder.get_head(tree[i])

    @staticmethod
    def collins_rule_NP(tree):
        for child in tree:
            if nltk_util.get_label(child) == "NP":
                return child

    @staticmethod
    def collins_rule_NML(tree):
        for child in tree:
            if nltk_util.get_label(child) == "NML":
                return child

    @staticmethod
    def collins_rule_PRN(tree):
        for child in tree:
            if nltk_util.get_label(child) == "PRN":
                return HeadFinder.get_head(child[0])

    @staticmethod
    def collins_rule_CD(tree):
        for i in range(len(tree)-1, -1, -1):
            if re.match("CD", nltk_util.get_label(tree[i])):
                return tree[i]

    @staticmethod
    def collins_rule_JJ(tree):
        for i in range(len(tree)-1, -1, -1):
            if re.match("JJ|JJS|RB", nltk_util.get_label(tree[i])):
                return tree[i]
            elif nltk_util.get_label(tree[i]) == "QP":
                return HeadFinder.get_head(tree[i])

    @staticmethod
    def collins_rule_last_word(tree):
        current_tree = tree[-1]
        while current_tree.height() > 2:
            current_tree = current_tree[-1]


    @staticmethod
    def adjust_head_for_nam(tokens, pos, type):
        # TODO: get rid of this ugly hack
        if len(pos) == 0:
            return Span(0, 0), "NOHEAD"

        start_regex = re.compile("")
        stop_regex = re.compile("CC|,|\.|:|;|V.*|IN|W.*|ADVP|NN$")

        if re.match("ORG.*|GPE.*|FAC.*|NORP.*|PRODUCT|EVENT|MONEY|WORK_OF_ART|LOC.*|LAW|LANGUAGE", type):
            start_regex = re.compile("NN(S)?|NNP(S)?")
            stop_regex = re.compile("V.*|IN|W.*|ADVP|,|-LRB-")
        elif type == "PERSON":
            start_regex = re.compile("NN(S)?|NNP(S)?")
            stop_regex = re.compile("IN|CC|,|\.|:|;|V.*|W.*|-LRB-")
        elif re.match("DATE|TIME", type):
            start_regex = re.compile("NN(S)?|NNP(S)?|CD")
        elif re.match("ORDINAL", type):
            start_regex = re.compile("NN|JJ|RB")
        elif re.match("CARDINAL", type):
            start_regex = re.compile("CD")
        elif re.match("QUANTITY|PERCENT", type):
            start_regex = re.compile("CD|JJ|NN")
        elif type == "NONE":
            start_regex = re.compile("NN(S)?|NNP(S)?|CD")
        else:
            raise Exception("Unknown named entity annotation: " + type)

        head_start = -1

        for i in range(0, len(tokens)):
            if head_start == -1 and start_regex.match(pos[i]):
                head_start = i
            elif head_start >= 0 and stop_regex.match(pos[i]):
                return Span(head_start, i-1), tokens[head_start:i]

        if head_start == -1:
            head_start = 0

        if pos[i] == "POS" and i == len(pos) - 1:
            i -= 1

        return Span(head_start, i), tokens[head_start:i+1]

    @staticmethod
    def rule_CC(tree):
        if nltk_util.get_label(tree) == "NP":
            for child in tree:
                if nltk_util.get_label(child) == "CC":
                    return child