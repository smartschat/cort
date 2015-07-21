""" Functions for extracting and filtering mentions in documents. """

from collections import defaultdict
import re

from cort.core import mentions
from cort.core import spans


__author__ = 'smartschat'


def extract_system_mentions(document, filter_mentions=True):
    """ Extract mentions from parse trees and named entity layers in a document.

    Args:
        document (ConLLDocument): The document from which mentions should be
            extracted.
        filter_mentions (bool): Indicates whether extracted mentions should
            be filtered. If set to True, filters:

                - mentions with the same head (retains one with largest span),
                - mentions whose head is embedded in another mention's head,
                - mentions whose head as POS tag JJ,
                - mentions of namend entity type QUANTITY, CARDINAL, ORDINAL,
                  MONEY or PERCENT,
                - mentions "mm", "hmm", "ahem", "um", "US" and "U.S.",
                - non-pronominal mentions embedded in appositions, and
                - pleonastic "it" and "you" detected via heuristics

    Returns:
        list(Mention): the sorted list of extracted system mentions. Includes a
        "dummy mention".
    """
    system_mentions = [mentions.Mention.from_document(span, document)
                       for span in __extract_system_mention_spans(document)]

    if filter_mentions:
        for post_processor in [
            post_process_same_head_largest_span,
            post_process_embedded_head_largest_span,
            post_process_by_head_pos,
            post_process_by_nam_type,
            post_process_weird,
            post_process_appositions,
            post_process_pleonastic_pronoun
        ]:
            system_mentions = post_processor(system_mentions)

    seen = set()

    # update set id and whether it is the first mention in gold entity
    for mention in system_mentions:
        mention.attributes["set_id"] = None

        annotated_set_id = mention.attributes["annotated_set_id"]

        mention.attributes["first_in_gold_entity"] = \
            annotated_set_id not in seen

        seen.add(annotated_set_id)

    system_mentions = [mentions.Mention.dummy_from_document(document)] \
        + system_mentions

    return system_mentions


def __extract_system_mention_spans(document):
    mention_spans = []
    for i, sentence_span in enumerate(document.sentence_spans):
        sentence_tree = document.parse[i]

        in_sentence_spans = __extract_mention_spans_for_sentence(
            sentence_tree,
            document.ner[sentence_span.begin:sentence_span.end+1])

        mention_spans += [spans.Span(sentence_span.begin + span.begin,
                                     sentence_span.begin + span.end)
                          for span in in_sentence_spans]

    return sorted(mention_spans)


def __extract_mention_spans_for_sentence(sentence_tree, sentence_ner):
    return sorted(list(set(
        [__get_in_tree_span(subtree) for subtree
            in sentence_tree.subtrees(__tree_filter)]
        + __get_span_from_ner(
            [pos[1] for pos in sentence_tree.pos()], sentence_ner)
    )))


def __extract_mention_spans_from_tree(sentence_tree):
    mention_spans = []

    for subtree in sentence_tree.subtrees(__tree_filter):
        mention_spans.append(__get_in_tree_span(subtree))

    return sorted(mention_spans)


def __tree_filter(tree):
    return tree.label() == "NP" or tree.label() == "PRP$"


def __get_span_from_ner(pos, ner):
    i = 0
    spans_from_ner = []
    while i < len(ner):
        current_tag = ner[i]
        if current_tag != "NONE":
            start = i

            while i+1 < len(ner) and ner[i+1] != "NONE" and ner[i] == ner[i+1]:
                i += 1

            if i+1 < len(pos) and pos[i+1] == "POS":
                i += 1

            spans_from_ner.append(spans.Span(start, i))

        i += 1

    return sorted(spans_from_ner)


def __get_in_tree_span(parented_tree):
    start = 0

    current_tree = parented_tree

    while current_tree.parent() is not None:
        for child in current_tree.parent():
            if child == current_tree:
                break
            else:
                start += len(child.leaves())

        current_tree = current_tree.parent()

    end = start + len(parented_tree.leaves()) - 1

    return spans.Span(start, end)


def post_process_by_head_pos(system_mentions):
    """ Removes mentions whose head has the part-of-speech tag JJ.

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    return sorted(
        [mention for mention
            in system_mentions
            if not re.match("^(JJ)",
                            mention.attributes["pos"][
                                mention.attributes["head_index"]])]
    )


def post_process_by_nam_type(system_mentions):
    """ Removes proper name mentions of types QUANTITY, CARDINAL, ORDINAL,
    MONEY and PERCENT.

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    return sorted(
        [mention for mention
            in system_mentions
            if mention.attributes["type"] != "NAM" or
            mention.attributes["ner"][mention.attributes["head_index"]] not in
            ["QUANTITY", "CARDINAL", "ORDINAL", "MONEY", "PERCENT"]]
    )


def post_process_weird(system_mentions):
    """ Removes all mentions which are "mm", "hmm", "ahem", "um", "US" or
    "U.S.".

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    return sorted(
        [mention for mention
         in system_mentions
         if " ".join(mention.attributes["tokens"]).lower() not in
         ["mm", "hmm", "ahem", "um"]
         and " ".join(mention.attributes["tokens"]) != "US"
         and " ".join(mention.attributes["tokens"]) != "U.S."]
    )


def post_process_pleonastic_pronoun(system_mentions):
    """ Removes pleonastic it and you.

    These are detected via the following heuristics:
        - it: appears in 'it _ _ that' or 'it _ _ _ that'
        - you: appears in 'you know'

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    filtered = []

    for mention in system_mentions:
        if " ".join(mention.attributes["tokens"]).lower() == "it":
            context_two = mention.get_context(2)
            context_three = mention.get_context(3)

            if context_two is not None:
                if context_two[-1] == "that":
                    continue

            if context_three is not None:
                if context_three[-1] == "that":
                    continue

        if " ".join(mention.attributes["tokens"]).lower() == "you":
            if mention.get_context(1) == ["know"]:
                continue

        filtered.append(mention)

    return sorted(filtered)


def post_process_same_head_largest_span(system_mentions):
    """ Removes a mention if there exists a larger mention with the same head.

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    head_span_to_mention = defaultdict(list)

    for mention in system_mentions:
        head_span_to_mention[mention.attributes["head_span"]].append(
            (mention.span.end - mention.span.begin, mention))

    return sorted([sorted(head_span_to_mention[head_span])[-1][1]
                   for head_span in head_span_to_mention])


def post_process_embedded_head_largest_span(system_mentions):
    """ Removes a mention its head is embedded in another head.

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    map_for_heads = {}

    for mention in system_mentions:
        head_span = mention.attributes["head_span"]
        if head_span.end not in map_for_heads:
            map_for_heads[head_span.end] = []

        map_for_heads[head_span.end].append(head_span.begin)

    post_processed_mentions = []

    for mention in system_mentions:
        head_span = mention.attributes["head_span"]
        head_begins = sorted(map_for_heads[head_span.end])
        if head_begins[0] < head_span.begin:
            continue
        else:
            post_processed_mentions.append(mention)

    return sorted(post_processed_mentions)


def post_process_appositions(system_mentions):
    """ Removes a mention its embedded in an apposition.

    Args:
        system_mentions (list(Mention): A list of system mentions.

    Returns:
        list(Mention): the filtered list of mentions.
    """
    appos = [mention for mention
             in system_mentions if mention.attributes["is_apposition"]]

    post_processed_mentions = []

    for mention in system_mentions:
        span = mention.span
        embedded_in_appo = False
        for appo in appos:
            appo_span = appo.span
            if appo_span.embeds(span) and appo_span != span:
                if len(appo.attributes["parse_tree"]) == 2:
                    embedded_in_appo = True
                elif (mention.attributes["parse_tree"] in
                        appo.attributes["parse_tree"]):
                    embedded_in_appo = True

        if mention.attributes["type"] == "PRO" or not embedded_in_appo:
            post_processed_mentions.append(mention)

    return sorted(post_processed_mentions)
