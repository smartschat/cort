""" Manage learning from training data and making predictions on test data. """


import codecs
import logging
import pickle
import random
import subprocess
import os

import cort


__author__ = 'smartschat'


def get_scores(output_filename, gold_filename, store_evaluation_scores=False):
    """ Evaluate coreference output against gold data using the CoNLL
    coreference scorer.

    Args:
        output_filename (str): The filename of the file that contains coreference output
            to evaluate.
        gold_filename (str): The filename of the file containing gold coreference information.
        store_evaluation_scores (bool): Whether evaluation scores should also be stored as
            files (to `output_filename`.METRIC_NAME, where METRIC_NAME iterates over muc,
            bcub and ceafe).

    Returns:
        str: A string containing the evaluation results, formatted as follows:
                        R       P       F1
                muc     68.68   76.72   72.48
                bcub    54.94   65.42   59.73
                ceafm   60.17   67.5    63.63
                ceafe   52.71   59.99   56.12
                blanc   56.52   65.51   60.58

                conll                   62.78

    """

    scorer_output = subprocess.check_output([
        "perl",
        cort.__path__[0] + "/reference-coreference-scorers/v8.01/scorer.pl",
        "all",
        gold_filename,
        os.getcwd() + "/" + output_filename]).decode()

    metrics = ['muc', 'bcub', 'ceafm', 'ceafe', 'blanc']

    metrics_results = {}

    metric = None

    results_formatted = ""

    buffer = []

    for line in scorer_output.split("\n"):
        if not line:
            continue

        splitted = line.split()

        if splitted[0] == "METRIC":
            if store_evaluation_scores and metric is not None:
                with open(output_filename + "." + metric, "w") as f:
                    f.write("\n".join(buffer))
                buffer = []

            metric = line.split()[1][:-1]

        if (metric != 'blanc' and line.startswith("Coreference:")) \
                or (metric == 'blanc' and line.startswith("BLANC:")):
            metrics_results[metric] = (
                float(splitted[5][:-1]),
                float(splitted[10][:-1]),
                float(splitted[12][:-1]),
            )

        if store_evaluation_scores:
            buffer.append(line)

    # missed last in loop above
    if store_evaluation_scores and metric is not None:
        with open(output_filename + "." + metric, "w") as f:
            f.write("\n".join(buffer))

    results_formatted += "\t\tR\tP\tF1\n"

    for metric in metrics:
        results_formatted += "\t" + metric + "\t" + \
                             "\t".join([str(val) for val in metrics_results[metric]]) + "\n"
    results_formatted += "\n"
    average = (metrics_results["muc"][2] + metrics_results["bcub"][2] +
               metrics_results["ceafe"][2]) / 3
    results_formatted += "\tconll\t\t\t" + format(average, '.2f') + "\n"

    return results_formatted


class TrainingConfiguration:
    """ Specify data, algorithms and parameters for learning a model.

    Attributes:
        corpus (Corpus): The corpus to learn from.
        instance_extractor (func): The function to extract training instances from the corpus.
        perceptron (Perceptron): The perceptron algorithm to learn parameters.
        n_iter (int): Number of iterations of the perceptron.
        random_seed (int): Random seed for shuffling the data during learning.
        output_name (str): The model learned in the ith iteration will be stored in the current
            directory to `output_name`+"-"+i+".obj"
    """
    def __init__(self,
                 corpus,
                 instance_extractor,
                 perceptron,
                 n_iter,
                 random_seed,
                 output_name):
        """ Initialize object that specifies data, algorithms and parameters for learning a model.

        Args:
            corpus (Corpus): The corpus to learn from.
            instance_extractor (func): The function to extract training instances from the corpus.
            perceptron (Perceptron): The perceptron algorithm to learn parameters.
            n_iter (int): Number of iterations of the perceptron.
            random_seed (int): Random seed for shuffling the data during learning.
            output_name (str): Filename template for storing models.
        """
        self.corpus = corpus
        self.instance_extractor = instance_extractor
        self.perceptron = perceptron
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.output_name = output_name


class TestingConfiguration:
    """ Specify data, algorithms and parameters for predicting coreference chains.

    Attributes:
        corpus (Corpus): The corpus to predict chains on.
        instance_extractor (func): The function to extract instances from the corpus.
        perceptron (Perceptron): The perceptron model to use.
        coref_extractor (func): The function to extract coreference chains from predictions.
        output_name (str): Filename template for storing output.
        gold_filename (str): Filename (including path) of the file that contains gold
            coreference annotation for evaluation.
    """
    def __init__(self,
                 corpus,
                 instance_extractor,
                 perceptron,
                 coref_extractor,
                 output_name,
                 gold_filename):
        self.corpus = corpus
        self.instance_extractor = instance_extractor
        self.perceptron = perceptron
        self.coref_extractor = coref_extractor
        self.output_name = output_name
        self.gold_filename = gold_filename


def learn(training_configuration, development_configuration=None):
    """ Learn a model for coreference resolution from training data.
    Optionally validate on development data.

    In particular, apply an instance/feature extractor to a training corpus and
    employ a machine learning model to learn a weight vector from these
    instances. Optionally evaluate on development data after each iteration.

    If `development_configuration` is not None, models and development output are
    stored on disk after each iteration. In iteration i, models are stored to
        training_configuration.output_name-i.obj
    and development output is stored to
        development_configuration.output_name-i.out
        development_configuration.output_name-i.antecedents

    If `development_configuration` is None, only the model of the last iteration is
    stored to
        training_configuration.output_name-i.obj

    Args:
        training_configuration (TrainingConfiguration): Configuration for the training
            setup.
        development_configuration (TestingConfiguration). Configuration for development
            setup. Defaults to None.

    Returns:
        The model learned in the last iteration. This is a tuple consisting of
            - **priors** (*dict(str,float)*): A prior weight for each label
              in the graphs representing the instances,
            - **weights** (*dict(str, array)*): A mapping of labels to weight
              vectors. For each label ``l``, ``weights[l]`` contains weights
              for each feature seen during training (for representing the
              features we employ *feature hashing*). If the graphs employed are
              not labeled, ``l`` is set to "+".
    """
    random.seed(training_configuration.random_seed)

    logging.info("Learning.")

    logging.info("\tExtracting instances and features.")
    training_substructures, training_arc_information = \
        training_configuration.instance_extractor.extract(
            training_configuration.corpus
        )

    if development_configuration:
        dev_substructures, dev_arc_information = \
            development_configuration.instance_extractor.extract(
                development_configuration.corpus
            )

    indices = list(range(len(training_substructures)))

    logging.info("\tFitting model parameters.")
    for i in range(1, training_configuration.n_iter + 1):
        logging.info("\t\tStarted epoch " + str(i))

        random.shuffle(indices)

        incorrect, decisions = training_configuration.perceptron.fit_one_epoch(
            training_substructures, training_arc_information,
            indices
        )
        logging.info("\t\tIncorrect predictions: " + str(incorrect) + "/" + str(decisions))

        if development_configuration:
            logging.info("\t\tValidation")
            model = training_configuration.perceptron.get_model()
            pickle.dump(model, open(training_configuration.output_name + "-" + str(i) + ".obj", "wb"),
                        protocol=2)
            development_configuration.perceptron.load_model(model)

            for doc in development_configuration.corpus:
                doc.antecedent_decisions = {}
                for mention in doc.system_mentions:
                    mention.attributes["antecedent"] = None
                    mention.attributes["set_id"] = None

            arcs, labels, scores = development_configuration.perceptron.predict(dev_substructures, dev_arc_information)
            coref_sets, antecedent_mapping = development_configuration.coref_extractor(
                arcs, labels, scores, development_configuration.perceptron.get_coref_labels())
            development_configuration.corpus.read_coref_decisions(coref_sets, antecedent_mapping)

            output_without_ending = development_configuration.output_name + "-" + str(i)

            development_configuration.corpus.write_to_file(codecs.open(output_without_ending + ".out", "w", "utf-8"))
            development_configuration.corpus.write_antecedent_decisions_to_file(
                codecs.open(output_without_ending + ".antecedents",
                            "w", "utf-8"))
            print(get_scores(output_without_ending + ".out", development_configuration.gold_filename, False))
        elif i == training_configuration.n_iter:
            model = training_configuration.perceptron.get_model()
            pickle.dump(model, open(training_configuration.output_name + "-" + str(i) + ".obj", "wb"),
                        protocol=2)

    return training_configuration.perceptron.get_model()


def predict(testing_configuration):
    """ According to a learned model, predict coreference information.

    Coreference output, antecedent decisions and evaluation metric results are
    stored to
        testing_configuration.output_name.out
        testing_configuration.output_name.antecedents
        testing_configuration.output_name.out.METRIC (METRIC in {muc, bcub and ceafe})
    respectively.

    Args:
        testing_configuration (TestingConfiguration). Configuration for testing
            setup.

    Returns:
        A tuple. The components are

            - **coref_sets** (*union_find.UnionFind*): An assignment of mentions
              to entities represented by a UnionFind data structure.
            - **antecedent_mapping** (*dict(Mention, Mention)*): A mapping of
              mentions to their antecedent.
    """
    logging.info("Predicting.")

    logging.info("\tRemoving coreference annotations from corpus.")
    for doc in testing_configuration.corpus:
        doc.antecedent_decisions = {}
        for mention in doc.system_mentions:
            mention.attributes["antecedent"] = None
            mention.attributes["set_id"] = None

    logging.info("\tExtracting instances and features.")
    substructures, arc_information = testing_configuration.instance_extractor.extract(
        testing_configuration.corpus)

    logging.info("\tDoing predictions.")
    arcs, labels, scores = testing_configuration.perceptron.predict(substructures,
                                                        arc_information)

    logging.info("\tClustering results.")
    union, antecedent_decisions = testing_configuration.coref_extractor(
        arcs, labels, scores, testing_configuration.perceptron.get_coref_labels())

    testing_configuration.corpus.read_coref_decisions(union, antecedent_decisions)

    testing_configuration.corpus.write_to_file(open(testing_configuration.output_name + ".out", "w"))
    testing_configuration.corpus.write_antecedent_decisions_to_file(open(testing_configuration.output_name + ".antecedents", "w"))

    if testing_configuration.gold_filename:
        print(get_scores(testing_configuration.output_name + ".out", testing_configuration.gold_filename, True))

    return union, antecedent_decisions
