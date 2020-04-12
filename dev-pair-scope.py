#!/usr/bin/env python

import codecs
import logging

import itertools
import numpy
import pyximport


pyximport.install(setup_args={"include_dirs": numpy.get_include()})


from cort.core import corpora
from cort.core import mention_extractor
from cort.coreference.approaches import mention_pairs
from cort.coreference import clusterer
from cort.coreference import cost_functions
from cort.coreference import features
from cort.coreference import instance_extractors
from cort.coreference import experiments


__author__ = 'smartschat'


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')

TRAIN_DATA = "train.auto"
DEV_DATA = "dev.auto"
DEV_GOLD = "dev.gold"
N_ITER = 50

logging.info("Reading in data.")

training_corpus = corpora.Corpus.from_file(
    "training",
    codecs.open(TRAIN_DATA, "r", "utf-8"))

dev_corpus = corpora.Corpus.from_file(
    "dev",
    codecs.open(DEV_DATA, "r", "utf-8"))

logging.info("Extracting system mentions.")
for doc in training_corpus:
    doc.system_mentions = mention_extractor.extract_system_mentions(doc)
for doc in dev_corpus:
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

coref_extractors = [
    ("closest_first", clusterer.closest_first),
]

instance_extractor_functions = [
    ("mod_soon_per_ana", mention_pairs.extract_training_substructures_mod_soon_per_anaphor),
    ("mod_soon_per_doc", mention_pairs.extract_training_substructures_mod_soon_per_document)
]

for ce_tuple, ie_tuple in itertools.product(coref_extractors, instance_extractor_functions):
    ce_desc, ce_func = ce_tuple
    ie_desc, ie_func = ie_tuple

    output_name = ce_desc + "-" + ie_desc

    print()
    print(output_name.upper())
    print("".join(["-"]*len(output_name)))

    training_perceptron = mention_pairs.MentionPairsPerceptron(
        cost_scaling=0
    )

    dev_perceptron = mention_pairs.MentionPairsPerceptron(
        cost_scaling=0
    )

    training_instance_extractor = instance_extractors.InstanceExtractor(
        ie_func,
        mention_features,
        pairwise_features,
        cost_functions.null_cost,
        training_perceptron.get_labels()
    )

    dev_instance_extractor = instance_extractors.InstanceExtractor(
        mention_pairs.extract_testing_substructures,
        mention_features,
        pairwise_features,
        cost_functions.null_cost,
        dev_perceptron.get_labels()
    )

    training_configuration = experiments.TrainingConfiguration(
        training_corpus,
        training_instance_extractor,
        training_perceptron,
        N_ITER,
        23,
        output_name
    )

    development_configuration = experiments.TestingConfiguration(
        dev_corpus,
        dev_instance_extractor,
        dev_perceptron,
        ce_func,
        output_name,
        DEV_GOLD
    )

    experiments.learn(training_configuration, development_configuration)
