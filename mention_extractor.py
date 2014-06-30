from collections import defaultdict
import re
import nltk_util
from spans import Span


__author__ = 'smartschat'


def extract_mention_spans(sentence_tree, sentence_ner):
    return sorted(list(set(
        [get_in_tree_span(subtree) for subtree in sentence_tree.subtrees(tree_filter)]
        + get_span_from_ner([pos[1] for pos in sentence_tree.pos()], sentence_ner)
    )))


def extract_mention_spans_from_tree(sentence_tree):
    spans = []

    for subtree in sentence_tree.subtrees(tree_filter):
        spans.append(get_in_tree_span(subtree))

    return sorted(spans)


def tree_filter(tree):
    return nltk_util.get_label(tree) == "NP" or nltk_util.get_label(tree) == "PRP$"


def get_span_from_ner(pos, ner):
    i = 0
    spans = []
    while i < len(ner):
        current_tag = ner[i]
        if current_tag != "NONE":
            start = i

            while i+1 < len(ner) and ner[i+1] != "NONE" and ner[i] == ner[i+1]:
                i += 1

            if i+1 < len(pos) and pos[i+1] == "POS":
                i += 1

            spans.append(Span(start, i))

        i += 1


    return sorted(spans)


def get_in_tree_span(parented_tree):
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

    return Span(start, end)


def post_process_mentions(mentions):
    same_head_largest_span = post_process_same_head_largest_span(mentions)
    embedded_head_largest_span = post_process_embedded_head_largest_span(same_head_largest_span)

    filtered_by_head_pos = [mention for mention in embedded_head_largest_span if not re.match("^(JJ)", mention.attributes["pos"][mention.attributes["head_index"]])]
    filtered_by_nam_type = [mention for mention in filtered_by_head_pos if mention.attributes["type"] != "NAM" or mention.attributes["ner"][mention.attributes["head_index"]] not in ["QUANTITY", "CARDINAL", "ORDINAL", "MONEY", "PERCENT"]]
    filter_weird_out = [mention for mention in filtered_by_nam_type if " ".join(mention.attributes["tokens"]).lower() not in ["mm", "hmm", "ahem", "um"] and " ".join(mention.attributes["tokens"]) != "US" and " ".join(mention.attributes["tokens"]) != "U.S."]
    filter_appositions = post_process_appositions(filter_weird_out)
    filter_pleonastic_pronoun = post_process_pleonastic_pronoun(filter_appositions)

    return sorted(filter_pleonastic_pronoun)


def post_process_pleonastic_pronoun(mentions):
    filtered = []

    for mention in mentions:
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

    return filtered


def post_process_same_head_largest_span(mentions):
    head_span_to_mention = defaultdict(list)

    for mention in mentions:
        head_span_to_mention[mention.attributes["head_span"]].append((mention.span.end - mention.span.begin, mention))

    return sorted([sorted(head_span_to_mention[head_span])[-1][1] for head_span in head_span_to_mention])


def post_process_embedded_head_largest_span(mentions):
    map_for_heads = {}

    for mention in mentions:
        head_span = mention.attributes["head_span"]
        if head_span.end not in map_for_heads:
            map_for_heads[head_span.end] = []

        map_for_heads[head_span.end].append(head_span.begin)

    post_processed_mentions = []

    for mention in mentions:
        head_span = mention.attributes["head_span"]
        head_begins = sorted(map_for_heads[head_span.end])
        if head_begins[0] < head_span.begin:
            continue
        else:
            post_processed_mentions.append(mention)

    return sorted(post_processed_mentions)


def post_process_appositions(mentions):
    appos = [mention for mention in mentions if mention.attributes["is_apposition"]]

    post_processed_mentions = []

    for mention in mentions:
        span = mention.span
        embedded_in_appo = False
        for appo in appos:
            appo_span = appo.span
            if appo_span.embeds(span) and appo_span != span:
                if len(appo.attributes["parse_tree"]) == 2:
                    embedded_in_appo = True
                elif mention.attributes["parse_tree"] in appo.attributes["parse_tree"]:
                    embedded_in_appo = True

        if mention.attributes["type"] == "PRO" or not embedded_in_appo:
            post_processed_mentions.append(mention)

    return sorted(post_processed_mentions)