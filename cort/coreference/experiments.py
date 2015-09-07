""" Manage learning from training data and making predictions on test data. """


import logging


__author__ = 'smartschat'


def learn(training_corpus, instance_extractor, perceptron):
    """ Learn a model for coreference resolution from training data.

    In particular, apply an instance/feature extractor to a training corpus and
    employ a machine learning model to learn a weight vector from these
    instances.

    Args:
        training_corpus (Corpus): The corpus to learn from.
        instance_extractor (InstanceExtracor): The instance extractor that
            defines the features and the structure of instances that are
            extracted during training.
        perceptron (Perceptron): A perceptron (including a decoder) that
            learns from the instances extracted by ``instance_extractor``.

    Returns:
        A tuple consisting of
            - **priors** (*dict(str,float)*): A prior weight for each label
              in the graphs representing the instances,
            - **weights** (*dict(str, array)*): A mapping of labels to weight
              vectors. For each label ``l``, ``weights[l]`` contains weights
              for each feature seen during training (for representing the
              features we employ *feature hashing*). If the graphs employed are
              not labeled, ``l`` is set to "+".
    """
    logging.info("Learning.")

    logging.info("\tExtracting instances and features.")
    substructures, arc_information = instance_extractor.extract(
        training_corpus)

    logging.info("\tFitting model parameters.")

    perceptron.fit(substructures, arc_information)

    return perceptron.get_model()


def predict(testing_corpus,
            instance_extractor,
            perceptron,
            coref_extractor):
    """ According to a learned model, predict coreference information.

    Args:
        testing_corpus (Corpus): The corpus to predict coreference on.
        instance_extractor (InstanceExtracor): The instance extracor that
            defines the features and the structure of instances that are
            extracted during testing.
        perceptron (Perceptron): A perceptron learned from training data.
        argmax_function (function): A decoder that computes the best-scoring
            coreference structure over a set of structures.
        coref_extractor (function): An extractor for consolidating pairwise
            predictions into coreference clusters.

    Returns:
        A tuple containing two dicts. The components are

            - **mention_entity_mapping** (*dict(Mention, int)*): A mapping of
              mentions to entity identifiers.
            - **antecedent_mapping** (*dict(Mention, Mention)*): A mapping of
              mentions to their antecedent (as determined by the
              ``coref_extractor``).
    """
    logging.info("Predicting.")

    logging.info("\tRemoving coreference annotations from corpus.")
    for doc in testing_corpus:
        doc.antecedent_decisions = {}
        for mention in doc.system_mentions:
            mention.attributes["antecedent"] = None
            mention.attributes["set_id"] = None

    logging.info("\tExtracting instances and features.")
    substructures, arc_information = instance_extractor.extract(testing_corpus)

    logging.info("\tDoing predictions.")
    arcs, labels, scores = perceptron.predict(substructures, arc_information)

    logging.info("\tClustering results.")

    return coref_extractor(arcs, labels, scores, perceptron.get_coref_labels())
