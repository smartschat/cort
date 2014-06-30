import re
import sys
from nltk import ParentedTree
from external_data import GenderData
from nltk.corpus import wordnet as wn
import head_finders
import nltk_util
from spans import Span

__author__ = 'smartschat'


def compute_number(attributes):
    number = "UNKNOWN"
    head_index = attributes["head_index"]
    pos = attributes["pos"][head_index]

    if attributes["type"] == "PRO":
        if attributes["citation_form"] in ["i", "you", "he", "she", "it"]:
            number = "SINGULAR"
        else:
            number = "PLURAL"
    elif attributes["type"] == "DEM":
        if attributes["head"][0].lower() in ["this", "that"]:
            number = "SINGULAR"
        else:
            number = "PLURAL"
    elif attributes["type"] in ["NOM", "NAM"]:
        if pos == "NNS" or pos == "NNPS":
            number = "PLURAL"
        else:
            number = "SINGULAR"

    if pos == "CC":
        number = "PLURAL"

    return number


def compute_gender(attributes):
    gender = "NEUTRAL"
    head_index = attributes["head_index"]
    gender_data = GenderData.Instance()

    if compute_number(attributes) == "PLURAL":
        gender = "PLURAL"
    elif attributes["type"] == "PRO":
        if attributes["citation_form"] == "he":
            gender = "MALE"
        elif attributes["citation_form"] == "she":
            gender = "FEMALE"
        elif attributes["citation_form"] == "it":
            gender = "NEUTRAL"
        elif attributes["citation_form"] in ["you", "we", "they"]:
            gender = "PLURAL"
    elif attributes["type"] == "NAM":
        if re.match(r"^mr(\.)?$", attributes["tokens"][0].lower()):
            gender = "MALE"
        elif re.match(r"^(miss|ms|mrs)(\.)?$", attributes["tokens"][0].lower()):
            gender = "FEMALE"
        elif not re.match(r"(PERSON|NONE)", attributes["ner"][head_index]):
            gender = "NEUTRAL"
        elif gender_data.look_up(attributes):
            gender = gender_data.look_up(attributes)
    elif attributes["type"] == "NOM":
        if wordnet_lookup_gender(" ".join(attributes["head"])):
            gender = wordnet_lookup_gender(" ".join(attributes["head"]))
        elif gender_data.look_up(attributes):
            gender = gender_data.look_up(attributes)

    if gender == "NEUTRAL" and compute_semantic_class(attributes) == "PERSON":
        gender = "UNKNOWN"

    return gender


def compute_semantic_class(attributes):
    semantic_class = "UNKNOWN"
    head_index = attributes["head_index"]

    if attributes["type"] == "PRO":
        if attributes["citation_form"] in ["i", "you", "he", "she", "we"]:
            semantic_class = "PERSON"
        elif attributes["citation_form"] == "they":
            semantic_class = "UNKNOWN"
        elif attributes["citation_form"] == "it":
            semantic_class = "OBJECT"
    elif attributes["type"] == "DEM":
        semantic_class = "OBJECT"
    elif attributes["ner"][head_index] != "NONE":
        ner_tag = attributes["ner"][head_index]
        if ner_tag == "PERSON":
            semantic_class = "PERSON"
        elif re.match("DATE|TIME|NUMBER|QUANTITY|MONEY|PERCENT", ner_tag):
            semantic_class = "NUMERIC"
        else:
            semantic_class = "OBJECT"
    # wordnet lookup
    elif attributes["type"] == "NOM" and wordnet_lookup_semantic_class(" ".join(attributes["head"])):
        semantic_class = wordnet_lookup_semantic_class(" ".join(attributes["head"]))

    return semantic_class


def wordnet_lookup_semantic_class(head):
    synsets = wn.synsets(head)

    while synsets:
        if sys.version_info[0] == 2:
            lemma_name = synsets[0].lemma_names[0]
        else:
            lemma_name = synsets[0].lemma_names()[0]

        if lemma_name == "person":
            return "PERSON"
        elif lemma_name == "object":
            return "OBJECT"

        synsets = synsets[0].hypernyms()


def wordnet_lookup_gender(head):
    synsets = wn.synsets(head)

    while synsets:
        if sys.version_info[0] == 2:
            lemma_name = synsets[0].lemma_names[0]
        else:
            lemma_name = synsets[0].lemma_names()[0]

        if lemma_name == "man" or lemma_name == "male":
            return "MALE"
        elif lemma_name == "woman" or lemma_name == "female":
            return "FEMALE"
        elif lemma_name == "person":
            return
        elif lemma_name == "entity":
            return "NEUTRAL"

        synsets = synsets[0].hypernyms()


def tree_is_apposition(tree):
    if nltk_util.get_label(tree) == "NP" and len(tree) > 1:
        if len(tree) == 2:
            return nltk_util.get_label(tree[0]) == "NP" and nltk_util.get_label(tree[1]) == "NP" and head_pos_starts_with(tree[1], "NNP")
        elif len(tree) == 3:
            return nltk_util.get_label(tree[0]) == "NP" and nltk_util.get_label(tree[1]) == "," and nltk_util.get_label(tree[2]) == "NP" and any_child_head_starts_with(tree, "NNP") and "DT" in set([child.pos()[0][1] for child in tree])
        elif len(tree) == 4:
            return nltk_util.get_label(tree[0]) == "NP" and nltk_util.get_label(tree[1]) == "," and nltk_util.get_label(tree[2]) == "NP" and nltk_util.get_label(tree[3]) == "," and any_child_head_starts_with(tree, "NNP") and "DT" in set([child.pos()[0][1] for child in tree])


def any_child_head_starts_with(tree, pos_tag):
    for child in tree:
        if head_pos_starts_with(child, pos_tag):
            return True

    return False


def head_pos_starts_with(tree, pos_tag):
    return head_finders.HeadFinder.get_head(tree).pos()[0][1].startswith(pos_tag)


def compute_head(mention_subtree, span, attributes):
        head_index = 0
        head = [attributes["tokens"][0]]

        if len(mention_subtree.leaves()) == len(attributes["tokens"]):
            head_tree = head_finders.HeadFinder.get_head(mention_subtree)
            head_index = get_head_index(head_tree, mention_subtree.pos())
            head = [head_tree[0]]

        head_span = Span(span.begin + head_index, span.begin + head_index)

        if attributes["pos"][head_index].startswith("NNP"):
            in_mention_span, head = head_finders.HeadFinder.adjust_head_for_nam(attributes["tokens"], attributes["pos"], attributes["ner"][head_index])
            head_span = Span(span.begin + in_mention_span.begin, span.begin + in_mention_span.end)

        head_index = head_span.end - span.begin

        if tree_is_apposition(mention_subtree):
            if len(mention_subtree) == 2:
                head_tree = mention_subtree[1]
                head = head_tree.leaves()
                head_index = span.end - span.begin
                head_span = Span(span.begin + len(mention_subtree[0].leaves()), span.end)
            else:
                start = 0
                for child in mention_subtree:
                    if head_pos_starts_with(child, "NNP"):
                        end = min([start + len(child.leaves()), len(attributes["tokens"])])
                        head_index = end - 1
                        in_mention_span, head = head_finders.HeadFinder.adjust_head_for_nam(attributes["tokens"][start:end], attributes["pos"][start:end], attributes["ner"][head_index])
                        head_span = Span(span.begin + in_mention_span.begin, span.begin + in_mention_span.end)
                        break
                    start += len(child.leaves())

        return head, head_span, head_index


def get_relevant_subtree(span, document):
    in_sentence_ids = document.in_sentence_ids[span.begin:span.end+1]
    in_sentence_span = Span(in_sentence_ids[0], in_sentence_ids[-1])
    sentence_tree = ParentedTree(document.get_parse(document.get_embedding_sentence(span)))
    spanning_leaves = sentence_tree.treeposition_spanning_leaves(in_sentence_span.begin, in_sentence_span.end+1)
    mention_subtree = sentence_tree[spanning_leaves]

    if mention_subtree in sentence_tree.leaves():
        mention_subtree = sentence_tree[spanning_leaves[:-2]]

    return mention_subtree


def get_grammatical_function(tree):
    parent = tree.parent()

    if parent is None:
        return "OTHER"
    else:
        parent_label = nltk_util.get_label(parent)

        if re.match(r"^(S|FRAG)", parent_label):
            return "SUBJECT"
        elif re.match(r"VP", parent_label):
            return "OBJECT"
        else:
            return "OTHER"


def get_head_index(head_with_pos, all_leaves):
    head_index = -1

    for i in range(0,len(all_leaves)):
        if head_with_pos[0] == all_leaves[i][0]:
            head_index = i

    return head_index


def get_type(pos, head_ner):
    if pos.startswith("NNP"):
        return "NAM"
    elif head_ner != "NONE":
        return "NAM"
    elif pos.startswith("PRP"):
        return "PRO"
    elif pos.startswith("DT"):
        return "DEM"
    elif pos.startswith("VB"):
        return "VRB"
    elif pos.startswith("NN"):
        return "NOM"
    else:
        return "NOM"


def get_fine_type(type, start_token, start_pos):
    if type == "NOM":
        if re.match("^(the|this|that|these|those|my|your|his|her|its|our|their)$", start_token.lower()):
            return "DEF"
        elif re.match("^NNP$", start_pos): # also matches NNPS!
            return "DEF"
        else:
            return "INDEF"
    elif type == "PRO":
        if re.match("^(i|you|he|she|it|we|they)$", start_token.lower()):
            return "PERS_NOM"
        elif re.match("^(me|you|him|her|it|us|you|them)$", start_token.lower()):
            return "PERS_ACC"
        elif re.match("^(myself|yourself|yourselves|himself|herself|itself|ourselves|themselves)$", start_token.lower()):
            return "REFL"
        elif start_pos == "PRP" and re.match("^(mine|yours|his|hers|its|ours|theirs|)$", start_token.lower()):
            return "POSS"
        elif start_pos == "PRP$" and re.match("^(my|your|his|her|its|our|their)$", start_token.lower()):
            return "POSS_ADJ"


def get_citation_form(pronoun):
    pronoun = pronoun.lower()
    if re.match("^(he|him|himself|his)$",  pronoun):
        return "he"
    elif re.match("^(she|her|herself|hers|her)$",  pronoun):
        return "she"
    elif re.match("^(it|itself|its)$",  pronoun):
        return "it"
    elif re.match("^(they|them|themselves|theirs|their)$",  pronoun):
        return "they"
    elif re.match("^(i|me|myself|mine|my)$",  pronoun):
        return "i"
    elif re.match("^(you|yourself|yourselves|yours|your)$",  pronoun):
        return "you"
    elif re.match("^(we|us|ourselves|ours|our)$",  pronoun):
        return "we"