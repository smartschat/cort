import argparse
import logging
import pickle
import sys

from cort.core import mention_extractor
from cort.coreference import experiments
from cort.coreference import features
from cort.singleton import singleton_instance_extractor
from cort.singleton import singleton_perceptron


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')


def train_models(training_corpus, part_id):

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
 
    
    perceptron = singleton_perceptron.SingletonPerceptron(
        cost_scaling=100,
        n_iter=5,
        seed=23
    )

    singleton_extractor = singleton_instance_extractor.InstanceExtractor(
    mention_features,
    general_features,
    perceptron.get_labels()
    )
    
    i=0
    for doc in training_corpus:
        if (part_id == "even" and i%2 == 0) or (part_id == "odd" and i%2==1) or (part_id == "all"):
            doc.system_mentions = mention_extractor.extract_system_mentions(doc)
        else:
            doc.system_mentions = []
        i+=1
    
    model = experiments.learn(
        training_corpus,
        singleton_extractor,
        perceptron
    )
    
    logging.info("Writing the singleton model to file singleton_train_"+ str(part_id) +".model.")
    pickle.dump(model, open(str("singleton_train_"+part_id+".model"), "wb"), protocol=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Train singleton models '
                                                 'models.')
    parser.add_argument('-train',
                        required=True,
                        dest='train_filename',
                        help='The training file. Must follow the format of the '
                             'CoNLL shared tasks on coreference resolution '
                             '(see http://conll.cemantix.org/2012/data.html).)')
    parser.add_argument('-n_iter',
                        dest='n_iter',
                        default=5,
                        help='Number of perceptron iterations. Defaults to 5.')
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

    return parser.parse_args()
    
