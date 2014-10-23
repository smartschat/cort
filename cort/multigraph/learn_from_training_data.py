from collections import defaultdict
import logging
import os
import pickle

from cort.core import util
from cort.multigraph import features
from cort.multigraph import multigraphs
from cort.multigraph import weighting_functions


__author__ = 'smartschat'

negative_features = [features.not_modifier,
                     features.not_compatible,
                     features.not_embedding,
                     features.not_speaker,
                     features.not_pronoun_distance,
                     features.not_anaphoric,
                     features.not_singleton]
positive_features = [features.alias,
                     features.non_pronominal_string_match,
                     features.head_match,
                     features.pronoun_same_canonical_form,
                     features.anaphor_pronoun,
                     features.speaker,
                     features.antecedent_is_subject,
                     features.antecedent_is_object,
                     features.substring,
                     features.lexical]

cmc = multigraphs.CorefMultigraphCreator(
    positive_features,
    negative_features,
    weighting_functions.for_each_relation_with_distance,
    {})

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

coreferent_pairs = defaultdict(int)
non_coreferent_pairs = defaultdict(int)

folder = "resources/corpora/conll/training/"
documents = []

for doc_file in sorted(os.listdir(folder)):
    doc = pickle.load(open(folder + doc_file, "rb"))
    documents.append(doc)
    print(doc)

    mentions = doc.system_mentions

    graph = cmc.construct_graph_from_mentions(mentions)

    for anaphor in graph.edges:
        for antecedent in graph.edges[anaphor]:
            if (graph.edges[anaphor][antecedent]["positive_relations"] and
                    not graph.edges[anaphor][antecedent][
                        "negative_relations"]):
                anaphor_cleaned = " ".join(
                    util.clean_via_pos(
                        anaphor.attributes["tokens"],
                        anaphor.attributes["pos"]))
                antecedent_cleaned = " ".join(
                    util.clean_via_pos(
                        antecedent.attributes["tokens"],
                        antecedent.attributes["pos"]))

                sorted_tuple = tuple(
                    sorted((anaphor_cleaned, antecedent_cleaned)))

                if doc.are_coreferent(anaphor, antecedent):
                    coreferent_pairs[sorted_tuple] += 1
                else:
                    non_coreferent_pairs[sorted_tuple] += 1


gender = {}
multiple = set()

for doc_file in sorted(os.listdir("resources/corpora/conll/training/")):
    print(doc_file)

    doc = pickle.load(
        open("resources/corpora/conll/training/" + doc_file, "rb"))

    mentions = doc.system_mentions

    for i in range(0, len(mentions)-1):
        antecedent = mentions[i]
        for j in range(i+1, len(mentions)):
            anaphor = mentions[j]

            if (anaphor.document.are_coreferent(anaphor, antecedent) and
                anaphor.attributes["type"] == "PRO" and
                    antecedent.attributes["type"] != "PRO"):
                citation_form = anaphor.attributes["citation_form"]
                if citation_form == "he":
                    gender[tuple(antecedent.attributes["head"])] = "MALE"
                elif citation_form == "she":
                    gender[tuple(antecedent.attributes["head"])] = "FEMALE"
                elif citation_form == "it":
                    gender[tuple(antecedent.attributes["head"])] = "NEUTRAL"

non_singletons = set()
singletons = {}

for doc_file in sorted(os.listdir("resources/corpora/conll/training/")):
    print(doc_file)
    print(len(non_singletons))

    doc = pickle.load(
        open("resources/corpora/conll/training/" + doc_file, "rb"))

    mentions = doc.system_mentions

    for i in range(0, len(mentions)):
        mention = mentions[i]
        mention_cleaned = " ".join(mention.attributes["tokens"])
        if mention_cleaned in non_singletons:
            continue
        else:
            if mention.attributes["annotated_set_id"]:
                non_singletons.add(mention_cleaned)
                if mention in singletons:
                    singletons.pop(mention_cleaned)
            else:
                if mention_cleaned not in singletons:
                    singletons[mention_cleaned] = 0

                singletons[mention_cleaned] += 1
