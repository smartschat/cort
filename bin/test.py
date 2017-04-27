#!/usr/bin/env python


from __future__ import print_function
import argparse
import codecs
import logging
import os
import pickle
import subprocess
import sys


import cort
from cort.core import corpora
from cort.core import mention_extractor
from cort.coreference import cost_functions
from cort.coreference import experiments
from cort.coreference import features
from cort.coreference import instance_extractors
from cort.util import import_helper
from cort.singleton import singleton_perceptron
from cort.singleton import singleton_instance_extractor


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')


def parse_args():
    parser = argparse.ArgumentParser(description='Predict coreference '
                                                 'relations.')
    parser.add_argument('-in',
                        required=True,
                        dest='input_filename',
                        help='The input file. Must follow the format of the '
                             'CoNLL shared tasks on coreference resolution '
                             '(see http://conll.cemantix.org/2012/data.html).)')
    parser.add_argument('-model',
                        required=True,
                        dest='model',
                        help='The model learned via cort-train.')
    parser.add_argument('-out',
                        dest='output_filename',
                        required=True,
                        help='The output file the predictions will be stored'
                             'in (in the CoNLL format.')
    parser.add_argument('-ante',
                        dest='ante',
                        help='The file where antecedent predictions will be'
                             'stored to.')
    parser.add_argument('-extractor',
                        dest='extractor',
                        default='cort.coreference.approaches.mention_ranking.extract_substructures',
                        help='The function to extract instances.')
    parser.add_argument('-perceptron',
                        dest='perceptron',
                        default='cort.coreference.approaches.mention_ranking.RankingPerceptron',
                        help='The perceptron to use.')
    parser.add_argument('-clusterer',
                        dest='clusterer',
                        default='cort.coreference.clusterer.all_ante',
                        help='The clusterer to use.')
    parser.add_argument('-gold',
                        dest='gold',
                        help='Gold data (in the CoNLL format) for evaluation.')
    parser.add_argument('-features',
                        dest='features',
                        help='The file containing the list of features. If not'
                             'provided, defaults to a standard set of'
                             'features.')

    return parser.parse_args()


def get_scores(output_data, gold_data):
    scorer_output = subprocess.check_output([
        "perl",
        cort.__path__[0] + "/reference-coreference-scorers/lea/scorer.pl",
        "all",
        gold_data,
        os.getcwd() + "/" + output_data,
        "none"]).decode()

    metrics = ['muc', 'bcub', 'ceafm', 'ceafe', 'lea']

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


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')

if sys.version_info[0] == 2:
    logging.warning("You are running cort under Python 2. cort is much more "
                    "efficient under Python 3.3+.")
args = parse_args()

if args.features:
    mention_features, pairwise_features = import_helper.get_features(
        args.features)
else:
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
        features.ancestry,
        features.singleton_score
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
        features.token_distance,
        features.genre
    ]

logging.info("Loading model.")
priors, weights = pickle.load(open(args.model, "rb"))

perceptron = import_helper.import_from_path(args.perceptron)(
    priors=priors,
    weights=weights,
    cost_scaling=0
)

extractor = instance_extractors.InstanceExtractor(
    import_helper.import_from_path(args.extractor),
    mention_features,
    pairwise_features,
    cost_functions.null_cost,
    perceptron.get_labels()
)

logging.info("Reading in data.")
testing_corpus = corpora.Corpus.from_file(
    "testing",
    codecs.open(args.input_filename, "r", "utf-8"))

logging.info("Extracting system mentions.")
for doc in testing_corpus:
    doc.system_mentions = mention_extractor.extract_system_mentions(doc)

if features.singleton_score in mention_features:
    
    singleton_mention_features = [
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
        features.pre_pre_token,
        features.preceding_token_pos,
        features.pre_pre_token_pos,
        features.next_token,
        features.next_next_token,
        features.next_token_pos,
        features.next_next_token_pos,
        features.governor,
        features.ancestry
    ]
    
    general_features = [
        features.has_exact_match,
        features.has_head_match,    
    ]
    
    if not os.path.isfile('singleton_train_all.model'):
        logging.info("singleton_train_all.model does not exist. \nPlease train the singleton models first.")
        sys.exit()
    logging.info("Loading singleton model.")
    priors, weights = pickle.load(open('singleton_train_all.model', 'rb'))
    
    singleton_percept = singleton_perceptron.SingletonPerceptron(
        priors=priors,
        weights=weights,
        cost_scaling=0
    )
    
    logging.info("\tExtracting singleton instances and features.")
    singleton_extractor = singleton_instance_extractor.InstanceExtractor(
        singleton_mention_features,
        general_features,
        singleton_percept.get_labels()

    )

    substructures, arc_information = singleton_extractor.extract(testing_corpus)

    tp, tn, fp, fn = 0, 0, 0, 0

    for doc in testing_corpus:
        singleton_percept.set_singleton_scores(doc, arc_information)
        for mention in doc.system_mentions[1:]:
            if mention.attributes["singletonScore"] > 0:
                if mention.attributes['annotated_set_id'] is not None:
                    tp +=1
                else:
                    fp +=1
 
            else:
                if mention.attributes['annotated_set_id'] is not None:
                    fn +=1
                else:
                    tn +=1
        
    recall = tp/(tp+fn)
    precision= tp/(tp+fp)
    F1= (2*recall*precision)/(recall+precision)

    print("Non-coreferent mention detection scores on the test set:")
    print("Recall=" + str(recall) + " Precision=" + str(precision) + " F1=" + str(F1))
    
mention_entity_mapping, antecedent_mapping = experiments.predict(
    testing_corpus,
    extractor,
    perceptron,
    import_helper.import_from_path(args.clusterer)
)

testing_corpus.read_coref_decisions(mention_entity_mapping, antecedent_mapping)


logging.info("Write corpus to file.")
testing_corpus.write_to_file(codecs.open(args.output_filename, "w", "utf-8"))

if args.ante:
    logging.info("Write antecedent decisions to file")
    testing_corpus.write_antecedent_decisions_to_file(open(args.ante, "w"))

if args.gold:
    logging.info("Evaluate.")
    print(get_scores(args.output_filename, args.gold))

logging.info("Done.")
