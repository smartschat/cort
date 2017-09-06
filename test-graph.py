#!/usr/bin/env python

import codecs
import logging

import pickle
import numpy
import pyximport


pyximport.install(setup_args={"include_dirs": numpy.get_include()})


from cort.core import corpora
from cort.core import mention_extractor
from cort.coreference.approaches import graph_based
from cort.coreference import clusterer
from cort.coreference import cost_functions
from cort.coreference import features
from cort.coreference import instance_extractors
from cort.coreference import experiments


__author__ = 'smartschat'


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')

TRAIN_DATA = "train+dev.auto"
TEST_DATA = "test.auto"
TEST_GOLD = "test.gold"

logging.info("Reading in data.")

training_corpus = corpora.Corpus.from_file(
    "training",
    codecs.open(TRAIN_DATA, "r", "utf-8"))

test_corpus = corpora.Corpus.from_file(
    "test",
    codecs.open(TEST_DATA, "r", "utf-8"))

logging.info("Extracting system mentions.")
for doc in training_corpus:
    doc.system_mentions = mention_extractor.extract_system_mentions(doc)
for doc in test_corpus:
    doc.system_mentions = mention_extractor.extract_system_mentions(doc)


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

configs = [
    # desc, train_extr, test_extr, train_perc, test_perc, cost_func, cost_scaling, n_iter, coref_ex
    ("test-graph-per_ana", graph_based.extract_per_anaphor_substructures,
     graph_based.extract_per_anaphor_substructures, graph_based.GraphPerceptronPerAnaphor,
     graph_based.GraphPerceptronPerAnaphor, cost_functions.cost_based_on_consistency_31, 200, 9, clusterer.all_ante),
    ("test-graph-per_doc", graph_based.extract_per_document_substructures,
     graph_based.extract_per_document_substructures, graph_based.GraphPerceptronPerDocument,
     graph_based.GraphPerceptronPerDocument, cost_functions.cost_based_on_consistency_31, 1000, 50, clusterer.all_ante),
]

for config_tuple in configs:
    desc, train_extr, test_extr, train_perc, test_perc, cost_func, cost_scaling, n_iter, coref_ex = config_tuple

    print()
    print(desc.upper())
    print("".join(["-"]*len(desc)))

    training_perceptron = train_perc(
        cost_scaling=cost_scaling
    )

    test_perceptron = test_perc(
        cost_scaling=0
    )

    training_instance_extractor = instance_extractors.InstanceExtractor(
        train_extr,
        mention_features,
        pairwise_features,
        cost_func,
        training_perceptron.get_labels()
    )

    test_instance_extractor = instance_extractors.InstanceExtractor(
        test_extr,
        mention_features,
        pairwise_features,
        cost_functions.null_cost,
        test_perceptron.get_labels()
    )

    training_configuration = experiments.TrainingConfiguration(
        training_corpus,
        training_instance_extractor,
        training_perceptron,
        n_iter,
        23,
        desc
    )

    experiments.learn(training_configuration)

    model = pickle.load(open(desc + "-" + str(n_iter) + ".obj", "rb"))

    test_perceptron.load_model(model)

    testing_configuration = experiments.TestingConfiguration(
        test_corpus,
        test_instance_extractor,
        test_perceptron,
        coref_ex,
        desc,
        TEST_GOLD
    )

    experiments.predict(testing_configuration)
