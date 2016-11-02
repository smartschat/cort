#!/usr/bin/env python

import argparse
import codecs
import logging
import pickle
import sys
import os.path

from cort.core import corpora
from cort.core import mention_extractor
from cort.coreference import experiments
from cort.coreference import features
from cort.coreference import instance_extractors
from cort.util import import_helper
from cort.singleton import singleton_perceptron
from cort.singleton import singleton_instance_extractor
import train_singleton_models



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')


def parse_args():
    parser = argparse.ArgumentParser(description='Train coreference resolution '
                                                 'models.')
    parser.add_argument('-in',
                        required=True,
                        dest='input_filename',
                        help='The input file. Must follow the format of the '
                             'CoNLL shared tasks on coreference resolution '
                             '(see http://conll.cemantix.org/2012/data.html).)')
    parser.add_argument('-out',
                        dest='output_filename',
                        required=True,
                        help='The output file the learned model will be saved '
                             'to.')
    parser.add_argument('-extractor',
                        dest='extractor',
                        default='cort.coreference.approaches.mention_ranking.extract_substructures',
                        help='The function to extract instances.')
    parser.add_argument('-perceptron',
                        dest='perceptron',
                        default='cort.coreference.approaches.mention_ranking.RankingPerceptron',
                        help='The perceptron to use.')
    parser.add_argument('-cost_function',
                        dest='cost_function',
                        default='cort.coreference.cost_functions.cost_based_on_consistency',
                        help='The cost function to use.')
    parser.add_argument('-n_iter',
                        dest='n_iter',
                        default=5,
                        help='Number of perceptron iterations. Defaults to 6.')
    parser.add_argument('-cost_scaling',
                        dest='cost_scaling',
                        default=100,
                        help='Scaling factor of the cost function. Defaults '
                             'to 1')
    parser.add_argument('-random_seed',
                        dest='seed',
                        default=23,
                        help='Random seed for training data shuffling. '
                             'Defaults to 23.')
    parser.add_argument('-features',
                        dest='features',
                        help='The file containing the list of features. If not'
                             'provided, defaults to a standard set of'
                             'features.')

    return parser.parse_args()


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
        features.singleton_score,
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
        features.genre,
    ]

perceptron = import_helper.import_from_path(args.perceptron)(
    cost_scaling=int(args.cost_scaling),
    n_iter=int(args.n_iter),
    seed=int(args.seed)
)

extractor = instance_extractors.InstanceExtractor(
    import_helper.import_from_path(args.extractor),
    mention_features,
    pairwise_features,
    import_helper.import_from_path(args.cost_function),
    perceptron.get_labels()
)

logging.info("Reading in data.")
training_corpus = corpora.Corpus.from_file("training",
                                           codecs.open(args.input_filename,
                                                       "r", "utf-8"))
for doc in training_corpus:
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
        features.ancestry,
    ]
    
    general_features = [
        features.has_exact_match,
        features.has_head_match,
    ]
 
    if not os.path.isfile('singleton_train_even.model') or \
       not os.path.isfile('singleton_train_odd.model') or \
       not os.path.isfile('singleton_train_all.model'):
           
        logging.info("Training singletpon models.")
        train_singleton_models.train_models(training_corpus, "even")
        train_singleton_models.train_models(training_corpus, "odd")
        train_singleton_models.train_models(training_corpus, "all")

    logging.info("Loading singleton models.")
    e_priors, e_weights = pickle.load(open("singleton_train_even.model", "rb"))
    
    singleton_perceptron_even = singleton_perceptron.SingletonPerceptron(
        priors=e_priors,
        weights=e_weights,
        cost_scaling=0
    )
    
    o_priors, o_weights = pickle.load(open("singleton_train_odd.model", "rb"))
    
    singleton_perceptron_odd = singleton_perceptron.SingletonPerceptron(
        priors=o_priors,
        weights=o_weights,
        cost_scaling=0
    )   
    
    logging.info("\tExtracting singleton instances and features.")
    singleton_extractor = singleton_instance_extractor.InstanceExtractor(
        singleton_mention_features,
        general_features,
        singleton_perceptron_odd.get_labels()
    )

    substructures, arc_information = singleton_extractor.extract(training_corpus)

    i=0
    for doc in training_corpus:
        if (i%2 == 0):
            singleton_perceptron_odd.set_singleton_scores(doc, arc_information)
        else:
            singleton_perceptron_even.set_singleton_scores(doc, arc_information)
        i+=1

model = experiments.learn(training_corpus, extractor, perceptron)

pickle.dump(model, open(args.output_filename, "wb"), protocol=2)

logging.info("Done.")
