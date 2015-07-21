""" Compute attributes of mentions. """

import re

from nltk.corpus import wordnet as wn

from cort.core import external_data
from cort.core import head_finders
from cort.core import spans


__author__ = 'smartschat'


def compute_number(attributes):
    """ Compute the number of a mention.

    Args:
        attributes (dict(str, object)): Attributes of the mention, must contain
            values for "type", "head_index" and "pos".

    Returns:
        str: the number of the mention -- one of UNKNOWN, SINGULAR and PLURAL.
    """
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
    """ Compute the gender of a mention.

    Args:
        attributes (dict(str, object)): Attributes of the mention, must contain
            values for "type", "head", "head_index" and, if the mention is a
            pronoun, "citation_form".

    Returns:
        str: the number of the mention -- one of UNKNOWN, MALE, FEMALE,
            NEUTRAL and PLURAL.
    """
    gender = "NEUTRAL"
    head_index = attributes["head_index"]
    gender_data = external_data.GenderData.get_instance()

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
        elif re.match(r"^(miss|ms|mrs)(\.)?$",
                      attributes["tokens"][0].lower()):
            gender = "FEMALE"
        elif not re.match(r"(PERSON|NONE)", attributes["ner"][head_index]):
            gender = "NEUTRAL"
        elif gender_data.look_up(attributes):
            gender = gender_data.look_up(attributes)
    elif attributes["type"] == "NOM":
        if __wordnet_lookup_gender(" ".join(attributes["head"])):
            gender = __wordnet_lookup_gender(" ".join(attributes["head"]))
        elif gender_data.look_up(attributes):
            gender = gender_data.look_up(attributes)

    if gender == "NEUTRAL" and compute_semantic_class(attributes) == "PERSON":
        gender = "UNKNOWN"

    return gender


def compute_semantic_class(attributes):
    """ Compute the semantic class of a mention.

    Args:
        attributes (dict(str, object)): Attributes of the mention, must contain
            values for "type", "head", "head_index" and, if the mention is a
            pronoun, "citation_form".

    Returns:
        str: the semantic class of the mention -- one of PERSON, OBJECT,
        NUMERIC and UNKNOWN.
    """
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
    elif (attributes["type"] == "NOM" and
            __wordnet_lookup_semantic_class(" ".join(attributes["head"]))):
        semantic_class = __wordnet_lookup_semantic_class(
            " ".join(attributes["head"]))

    return semantic_class


def __wordnet_lookup_semantic_class(head):
    synsets = wn.synsets(head)

    while synsets:
        lemma_name = synsets[0].lemma_names()[0]

        if lemma_name == "person":
            return "PERSON"
        elif lemma_name == "object":
            return "OBJECT"

        synsets = synsets[0].hypernyms()


def __wordnet_lookup_gender(head):
    synsets = wn.synsets(head)

    while synsets:
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


def is_apposition(attributes):
    """ Compute whether the mention is an apposition, as in "Secretary of
    State Madeleine Albright" or "Barack Obama, the US president".

    Args:
        attributes (dict(str, object)): Attributes of the mention, must contain
            a value for "parse_tree".

    Returns:
        bool: Whether the mention is an apposition.
    """
    tree = attributes["parse_tree"]

    if tree.label() == "NP" and len(tree) > 1:
        if len(tree) == 2:
            return (tree[0].label() == "NP" and
                    tree[1].label() == "NP" and
                    __head_pos_starts_with(tree[1], "NNP"))
        elif len(tree) == 3:
            return (tree[0].label() == "NP" and
                    tree[1].label() == "," and
                    tree[2].label() == "NP" and
                    __any_child_head_starts_with(tree, "NNP") and
                    "DT" in set([child.pos()[0][1] for child in tree]))
        elif len(tree) == 4:
            return (tree[0].label() == "NP" and
                    tree[1].label() == "," and
                    tree[2].label() == "NP" and
                    tree[3].label() == "," and
                    __any_child_head_starts_with(tree, "NNP") and
                    "DT" in set([child.pos()[0][1] for child in tree]))


def __any_child_head_starts_with(tree, pos_tag):
    for child in tree:
        if __head_pos_starts_with(child, pos_tag):
            return True

    return False


def __head_pos_starts_with(tree, pos_tag):
    head_finder = head_finders.HeadFinder()
    return head_finder.get_head(tree).pos()[0][1].startswith(pos_tag)


def compute_head_information(attributes):
    """ Compute the head of the mention.

    Args:
        attributes (dict(str, object)): Attributes of the mention, must contain
            values for "tokens", "parse_tree", "pos", "ner", "is_apposition"

    Returns:
        (list(str), Span, int): The head, the head span (in the document) and
        the starting index of the head (in the mention).
    """
    mention_subtree = attributes["parse_tree"]

    head_finder = head_finders.HeadFinder()
    head_index = 0
    head = [attributes["tokens"][0]]

    if len(mention_subtree.leaves()) == len(attributes["tokens"]):
        head_tree = head_finder.get_head(mention_subtree)
        head_index = get_head_index(head_tree, mention_subtree.pos())
        head = [head_tree[0]]

    in_mention_span = spans.Span(head_index, head_index)

    if attributes["pos"][head_index].startswith("NNP"):
        in_mention_span, head = \
            head_finders.HeadFinder.adjust_head_for_nam(
                attributes["tokens"],
                attributes["pos"],
                attributes["ner"][head_index])

    # proper name mention: head index last word of head
    # (e.g. "Obama" in "Barack Obama")
    head_index = in_mention_span.end

    # special handling for appositions
    if attributes["is_apposition"]:
        # "Secretary of State Madeleine Albright"
        # => take "Madeleine Albright" as head
        if len(mention_subtree) == 2:
            head_tree = mention_subtree[1]
            head = head_tree.leaves()
            in_mention_span = spans.Span(
                len(mention_subtree[0].leaves()),
                len(attributes["tokens"]) - 1)
            head_index = in_mention_span.end
        else:
            start = 0
            for child in mention_subtree:
                if __head_pos_starts_with(child, "NNP"):
                    end = min(
                        [start + len(child.leaves()),
                         len(attributes["tokens"])])
                    head_index = end - 1
                    in_mention_span, head = \
                        head_finders.HeadFinder.adjust_head_for_nam(
                            attributes["tokens"][start:end],
                            attributes["pos"][start:end],
                            attributes["ner"][head_index])
                    break
                start += len(child.leaves())

    return head, in_mention_span, head_index


def get_relevant_subtree(span, document):
    """ Get the fragment of the parse tree and the input span.

    Args:
        span (Span): A span in a document.
        document (CoNLLDocument): A document.

    Returns:
        nltk.ParentedTree: The fragment of the parse tree at the span in the
        document.
    """
    in_sentence_ids = document.in_sentence_ids[span.begin:span.end+1]
    in_sentence_span = spans.Span(in_sentence_ids[0], in_sentence_ids[-1])

    sentence_id, sentence_span = document.get_sentence_id_and_span(span)

    sentence_tree = document.parse[sentence_id]

    spanning_leaves = sentence_tree.treeposition_spanning_leaves(
        in_sentence_span.begin, in_sentence_span.end+1)
    mention_subtree = sentence_tree[spanning_leaves]

    if mention_subtree in sentence_tree.leaves():
        mention_subtree = sentence_tree[spanning_leaves[:-2]]

    return mention_subtree


def get_grammatical_function(attributes):
    """ Compute the grammatical function of a mention in its sentence.

    Args:
        attributes (dict(str, object)): Attributes of the mention, must contain
            a value for "parse_tree".

    Returns:
        str: The grammatical function of the mention in its sentence, one of
        SUBJECT, OBJECT and OTHER.
    """
    tree = attributes["parse_tree"]
    parent = tree.parent()

    if parent is None:
        return "OTHER"
    else:
        parent_label = parent.label()

        if re.match(r"^(S|FRAG)", parent_label):
            return "SUBJECT"
        elif re.match(r"VP", parent_label):
            return "OBJECT"
        else:
            return "OTHER"


def get_head_index(head_with_pos, all_leaves):
    head_index = -1

    for i in range(0, len(all_leaves)):
        if head_with_pos[0] == all_leaves[i][0]:
            head_index = i

    return head_index


def get_type(attributes):
    """ Compute mention type.

    Args:
        attributes (dict(str, object)): Attributes of the mention, must contain
            values for "pos", "ner" and "head_index".

    Returns:
        str: The mention type, one of NAM (proper name), NOM (common noun),
        PRO (pronoun), DEM (demonstrative pronoun) and VRB (verb).
    """
    pos = attributes["pos"][attributes["head_index"]]
    head_ner = attributes["ner"][attributes["head_index"]]

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


def get_fine_type(attributes):
    """ Compute fine-grained mention type.

    Args:
        attributes (dict(str, object)): Attributes of the mention, must contain
            values for "type", "tokens" and "pos".

    Returns:
        str: The fine-grained mention type, one of

            - DEF (definite noun phrase),
            - INDEF (indefinite noun phrase),
            - PERS_NOM (personal pronoun, nominative case),
            - PERS_ACC (personal pronoun, accusative),
            - REFL (reflexive pronoun),
            - POSS (possessive pronoun),
            - POSS_ADJ (possessive adjective) or
            - None.
    """
    coarse_type = attributes["type"]
    start_token = attributes["tokens"][0]
    start_pos = attributes["pos"][0]

    if coarse_type == "NOM":
        if re.match("^(the|this|that|these|those|my|your|his|her|its|our|" +
                    "their)$", start_token.lower()):
            return "DEF"
        elif re.match("^NNP$", start_pos):  # also matches NNPS!
            return "DEF"
        else:
            return "INDEF"
    elif coarse_type == "PRO":
        if re.match("^(i|you|he|she|it|we|they)$", start_token.lower()):
            return "PERS_NOM"
        elif re.match("^(me|you|him|her|it|us|you|them)$",
                      start_token.lower()):
            return "PERS_ACC"
        elif re.match("^(myself|yourself|yourselves|himself|herself|itself|" +
                      "ourselves|themselves)$", start_token.lower()):
            return "REFL"
        elif start_pos == "PRP" and re.match("^(mine|yours|his|hers|its|" +
                                             "ours|theirs|)$",
                                             start_token.lower()):
            return "POSS"
        elif start_pos == "PRP$" and re.match("^(my|your|his|her|its|our|" +
                                              "their)$", start_token.lower()):
            return "POSS_ADJ"


def get_citation_form(attributes):
    """ Compute the citation form of a pronominal mention.

    Args:
        attributes (dict(str, object)): Attributes of the mention, must contain
            the key "tokens".

    Returns:
        str: The citation form of the pronoun, one of "i", "you", "he", "she",
        "it", "we", "they" and None.
    """
    pronoun = attributes["tokens"][0]

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
