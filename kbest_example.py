#!/usr/bin/env python


from __future__ import print_function

import numpy
import pyximport
from collections import defaultdict

pyximport.install(setup_args={"include_dirs": numpy.get_include()})

import codecs
import logging
import os
import pickle
import subprocess


import cort
from cort.core import corpora
from cort.core import mention_extractor
from cort.coreference import clusterer
from cort.coreference import cost_functions
from cort.coreference import features
from cort.coreference import instance_extractors
from cort.coreference.approaches import antecedent_trees


__author__ = 'smartschat'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')


def get_scores(output_data, gold_data):
    scorer_output = subprocess.check_output([
        "perl",
        cort.__path__[0] + "/reference-coreference-scorers/v8.01/scorer.pl",
        "all",
        gold_data,
        os.getcwd() + "/" + output_data,
        "none"]).decode()

    metrics = ['muc', 'bcub', 'ceafm', 'ceafe', 'blanc']

    metrics_results = {}

    metric = None

    results_formatted = ""

    for line in scorer_output.split("\n"):
        if not line:
            continue

        splitted = line.split()

        if splitted[0] == "METRIC":
            metric = line.split()[1][:-1]

        if (metric != 'blanc' and line.startswith("Coreference:")) \
           or (metric == 'blanc' and line.startswith("BLANC:")):
            metrics_results[metric] = (
                float(splitted[5][:-1]),
                float(splitted[10][:-1]),
                float(splitted[12][:-1]),
            )

    results_formatted += "\tR\tP\tF1\n"

    for metric in metrics:
        results_formatted += metric + "\t" + \
            "\t".join([str(val) for val in metrics_results[metric]]) + "\n"
    results_formatted += "\n"
    average = (metrics_results["muc"][2] + metrics_results["bcub"][2] +
               metrics_results["ceafe"][2])/3
    results_formatted += "conll\t\t\t" + format(average, '.2f') + "\n"

    return results_formatted


def get_upper_bound(output_data, gold_data, k):
    metrics = ["muc", "bcub", "ceafe"]

    doc_to_metric_to_f1 = {}
    average_f1 = defaultdict(float)
    doc_to_metric_to_rec_prec = {}

    results_formatted = "\tR\tP\tF1\n"

    # store results
    for i in range(k):
        for metric in metrics:
            scorer_output = subprocess.check_output([
                "perl",
                cort.__path__[0] + "/reference-coreference-scorers/v8.01/scorer.pl",
                metric,
                gold_data,
                os.getcwd() + "/" + output_data + "-" + str(i)]).decode()

            doc_identifier = None

            for line in scorer_output.split("\n"):
                if not line:
                    continue

                if line.startswith("("):
                    doc_identifier = line.strip()
                elif line.startswith("------"):
                    doc_identifier = None
                elif line.startswith("Recall"):
                    splitted = line.split()
                    f1 = float(splitted[-1].replace("%", ""))
                    doc_to_metric_to_f1[(doc_identifier, metric, i)] = f1
                    average_f1[(doc_identifier, i)] += f1
                    doc_to_metric_to_rec_prec[(doc_identifier, metric, i)] = (
                            float(splitted[1].replace("(", "")),
                            float(splitted[3].replace(")", "")),
                            float(splitted[6].replace("(", "")),
                            float(splitted[8].replace(")", "")),
                        )
    for key in average_f1:
        average_f1[key] /= 3


    # compute optimal position in list for each doc
    docs = sorted(list(set([key[0] for key in doc_to_metric_to_f1.keys()])))
    best_for_doc = []

    for doc in docs:
        arr = numpy.array([average_f1[(doc, i)] for i in range(k)])
        best_for_doc.append(arr.argmax())

    metric_to_f1 = {}

    # compute upper bound
    for metric in metrics:
        rec = sum([doc_to_metric_to_rec_prec[doc, metric, best_for_doc[i]][0] for i, doc in enumerate(docs)]) / sum(
            [doc_to_metric_to_rec_prec[doc, metric, best_for_doc[i]][1] for i, doc in enumerate(docs)])
        prec = sum([doc_to_metric_to_rec_prec[doc, metric, best_for_doc[i]][2] for i, doc in enumerate(docs)]) / sum(
            [doc_to_metric_to_rec_prec[doc, metric, best_for_doc[i]][3] for i, doc in enumerate(docs)])

        f1 = 2*prec*rec/(prec+rec)

        metric_to_f1[metric] = f1

        results_formatted += metric + "\t" + format(rec*100, '.2f') + "\t" + format(prec*100, '.2f') + "\t" + format(f1*100, '.2f') + "\n"

    average = sum(metric_to_f1.values())/3
    results_formatted += "\n"
    results_formatted += "conll\t\t\t" + format(average*100, '.2f') + "\n"

    return results_formatted


# input/output/k-best parameters

MODEL_FILE = "model-latent-train.obj"
CORPUS_FILE = "a2e_dev.conll"
OUTPUT_NAME = "output"
OUTPUT_ANTECEDENT_NAME = "antecedent"
GOLD_FILE = "a2e_dev.conll"
K = 10

# define  features
mention_features = [
    features.fine_type,
    features.gender,
    features.number,
    features.sem_class,
    features.deprel,
    features.head_ner,
    features.length,
    features.head,
    features.first,
    features.last,
    features.preceding_token,
    features.next_token,
    features.governor,
    features.ancestry
]

pairwise_features = [
    features.exact_match,
    features.head_match,
    features.same_speaker,
    features.alias,
    features.sentence_distance,
    features.embedding,
    features.modifier,
    features.tokens_contained,
    features.head_contained,
    features.token_distance
]

# load model, perceptron, instance extractors
logging.info("Loading model.")
priors, weights = pickle.load(open(MODEL_FILE, "rb"))

perceptron = antecedent_trees.AntecedentTreePerceptron(
    priors=priors,
    weights=weights,
    cost_scaling=0
)

extractor = instance_extractors.InstanceExtractor(
    antecedent_trees.extract_substructures,
    mention_features,
    pairwise_features,
    cost_functions.null_cost,
    perceptron.get_labels()
)

# read in and preprocess data
logging.info("Reading in data.")
testing_corpus = corpora.Corpus.from_file(
    "testing",
    codecs.open(CORPUS_FILE, "r", "utf-8"))

logging.info("Extracting system mentions.")
for doc in testing_corpus:
    doc.system_mentions = mention_extractor.extract_system_mentions(doc)

logging.info("Extracting instances and features.")
substructures, arc_information = extractor.extract(testing_corpus)

# perform predictions -- each k-best list contains arcs, arc labels (empty here)
# and scores for arcs
logging.info("Performing k-best predictions with k=" + str(K))
prediction_lists = perceptron.predict_kbest(substructures, arc_information, K)

for i, (arcs, labels, scores) in enumerate(prediction_lists):
    # closter coref info
    mention_entity_mapping, antecedent_mapping = clusterer.all_ante(
        arcs, labels, scores, perceptron.get_coref_labels())

    print(str(i+1) + "-best solution")

    # print sum of arc scores for each doc
    for doc_arcs, doc_scores in zip(arcs, scores):
        print(doc_arcs[0][0].document, sum(doc_scores))

    # store and write coref info
    for doc in testing_corpus:
        doc.antecedent_decisions = {}
        for mention in doc.system_mentions:
            mention.attributes["antecedent"] = None
            mention.attributes["set_id"] = None

    testing_corpus.read_coref_decisions(mention_entity_mapping,
                                        antecedent_mapping)

    logging.info("\tWrite corpus to file.")
    testing_corpus.write_to_file(codecs.open(OUTPUT_NAME + "-" + str(i),
                                             "w", "utf-8"))

    logging.info("\tWrite antecedent decisions to file")
    testing_corpus.write_antecedent_decisions_to_file(
    open(OUTPUT_ANTECEDENT_NAME + "-" + str(i), "w"))

    # finally evaluate
    logging.info("\tEvaluate.")
    print(get_scores(OUTPUT_NAME + "-" + str(i), GOLD_FILE))

logging.info("\tCompute upper bound.")
print(get_upper_bound(OUTPUT_NAME, GOLD_FILE, K))
