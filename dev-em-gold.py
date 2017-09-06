#!/usr/bin/env python

import codecs
import logging
import os
import random
import subprocess

import numpy
import pickle
import pyximport

import cort

pyximport.install(setup_args={"include_dirs": numpy.get_include()})


from cort.core import corpora
from cort.core import mention_extractor
from cort.coreference.approaches import entity, entity_left_to_right
from cort.coreference import clusterer
from cort.coreference import cost_functions
from cort.coreference import features
from cort.coreference import instance_extractors


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

    results_formatted += "\t\tR\tP\tF1\n"

    for metric in metrics:
        results_formatted += "\t" + metric + "\t" + \
            "\t".join([str(val) for val in metrics_results[metric]]) + "\n"
    results_formatted += "\n"
    average = (metrics_results["muc"][2] + metrics_results["bcub"][2] +
               metrics_results["ceafe"][2])/3
    results_formatted += "\tconll\t\t\t" + format(average, '.2f') + "\n"

    return results_formatted


mention_features = [
    features.fine_type,
    features.gender,
    features.number,
    features.sem_class,
    features.gr_func,
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

cluster_features = [
    features.cluster_size_antecedent_cluster,
]

dynamic_features = [
    features.dynamic_ante_has_ante,
]

training_perceptron = entity_left_to_right.EntityLeftToRightPerceptronWithGoldRollIn(
    cost_scaling=200,
    cluster_features=cluster_features,
    dynamic_features=dynamic_features,
)

dev_perceptron = entity_left_to_right.EntityLeftToRightPerceptronWithGoldRollIn(
    cost_scaling=0,
    cluster_features=cluster_features,
    dynamic_features=dynamic_features,
)

training_instance_extractor = instance_extractors.InstanceExtractor(
    entity.extract_substructures,
    mention_features,
    pairwise_features,
    cost_functions.cost_based_on_consistency_21,
    training_perceptron.get_labels()
)

dev_instance_extractor = instance_extractors.InstanceExtractor(
    entity.extract_substructures,
    mention_features,
    pairwise_features,
    cost_functions.null_cost,
    dev_perceptron.get_labels()
)

logging.info("Reading in data.")

training_corpus = corpora.Corpus.from_file(
    "training",
    codecs.open("train.auto", "r", "utf-8")
    #codecs.open("a2e_split1.conll", "r", "utf-8")
    #codecs.open("a2e_dev.conll", "r", "utf-8")
    #codecs.open("a2e_0001_part_00.v4_auto_conll", "r", "utf-8")
    #codecs.open("/data/nlp/martscsn/thesis/data/input/train.auto", "r", "utf-8")
)

dev_corpus = corpora.Corpus.from_file(
    "dev",
    codecs.open("dev.auto", "r", "utf-8")
    #codecs.open("a2e_split2.conll", "r", "utf-8")
    #codecs.open("a2e_0001_part_00.v4_auto_conll", "r", "utf-8")
    #codecs.open("/data/nlp/martscsn/thesis/data/input/dev.auto", "r", "utf-8")
)

logging.info("Extracting system mentions.")
for doc in training_corpus:
    doc.system_mentions = mention_extractor.extract_system_mentions(doc)
for doc in dev_corpus:
    doc.system_mentions = mention_extractor.extract_system_mentions(doc)

logging.info("\tExtracting instances and features.")
training_substructures, training_arc_information = training_instance_extractor.extract(training_corpus)
dev_substructures, dev_arc_information = dev_instance_extractor.extract(dev_corpus)

n_iter = 30
output_name = "l2s_em_gold"

random.seed(23)

indices = list(range(len(training_substructures)))

logging.info("\tEvaluating model.")
for i in range(1, n_iter+1):
    logging.info("\t\tNow epoch " + str(i))
    random.shuffle(indices)
    incorrect, decisions = training_perceptron.fit_one_epoch(
            training_substructures, training_arc_information,
            indices)
    logging.info("\t\tIncorrect predictions: " + str(incorrect) + "/" + str(decisions))
    model = training_perceptron.get_model()
    pickle.dump(model, open(output_name + "-" + str(i) + ".obj", "wb"),
                protocol=2)
    dev_perceptron.load_model(pickle.load(open(output_name + "-" + str(i) + ".obj", "rb")))

    arcs, labels, scores = dev_perceptron.predict(dev_substructures, dev_arc_information)

    for doc in dev_corpus:
        doc.antecedent_decisions = {}
        for mention in doc.system_mentions:
            mention.attributes["antecedent"] = []
            mention.attributes["set_id"] = None

    union, antecedent_mapping = clusterer.all_ante(arcs, labels, scores, dev_perceptron.get_coref_labels())
    dev_corpus.read_coref_decisions(union, antecedent_mapping)

    output_file = output_name + "-" + str(i) + ".out"

    dev_corpus.write_to_file(codecs.open(output_file, "w", "utf-8"))
    dev_corpus.write_antecedent_decisions_to_file(codecs.open(output_name + "-" + str(i) + ".antecedents", "w", "utf-8"))

    print(get_scores(output_file,
                     #"a2e_split2.conll",
                     #"a2e_0001_part_00.v4_auto_conll",
                     #"/data/nlp/martscsn/thesis/data/input/dev.gold"
                     "dev.gold"
                     )
          )

#experiment = experiments.Experiment(
#    training_instance_extractor,
#    training_perceptron,
#    dev_instance_extractor,
#    dev_perceptron,
#    clusterer.aggressive_merge
#)
#
#experiment.learn(training_corpus, dev_corpus, 20, "pair-vanilla",
#                 "/data/nlp/martscsn/thesis/data/input/dev.auto")

