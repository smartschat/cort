from __future__ import print_function


import subprocess


__author__ = 'smartschat'


def get_extractor(data_set, system):
    if system == "closest" or system == "latent":
        return "cort.coreference.approaches.mention_ranking.extract_substructures"
    elif system == "tree":
        return "cort.coreference.approaches.antecedent_trees.extract_substructures"
    elif system == "pair":
        if data_set == "train":
            return "cort.coreference.approaches.mention_pairs" \
                   ".extract_training_substructures"
        else:
            return "cort.coreference.approaches.mention_pairs" \
                   ".extract_testing_substructures"


def get_perceptron(system):
    if system == "pair":
        return "cort.coreference.approaches.mention_pairs.MentionPairsPerceptron"
    elif system == "closest":
        return "cort.coreference.approaches.mention_ranking.RankingPerceptronClosest"
    elif system == "latent":
        return "cort.coreference.approaches.mention_ranking.RankingPerceptron"
    elif system == "tree":
        return "cort.coreference.approaches.antecedent_trees.AntecedentTreePerceptron"


def get_cost_function(system):
    if system == "pair":
        return "cort.coreference.cost_functions.null_cost"
    else:
        return "cort.coreference.cost_functions.cost_based_on_consistency"


def get_clusterer(system):
    if system == "pair":
        return "cort.coreference.clusterer.best_first"
    else:
        return "cort.coreference.clusterer.all_ante"


systems = ["pair", "closest", "latent", "tree"]
data_sets = ["dev", "test"]

for system in systems:
    print("Training", system, "on train.")
    subprocess.call([
        "cort-train",
        "-in", "/data/nlp/martscsn/thesis/data/input/train.auto",
        "-out", "model-" + system + "-train.obj",
        "-extractor", get_extractor("train", system),
        "-perceptron", get_perceptron(system),
        "-cost_function", get_cost_function(system),
        "-cost_scaling", "100"])

    print("Training", system, "on dev+train.")
    subprocess.call([
        "cort-train",
        "-in", "/data/nlp/martscsn/thesis/data/input/train+dev.auto",
        "-out", "model-" + system + "-train+dev.obj",
        "-extractor", get_extractor("train", system),
        "-perceptron", get_perceptron(system),
        "-cost_function", get_cost_function(system),
        "-cost_scaling", "100"])

    for data_set in data_sets:
        print("Predicting", system, "on", data_set)
        if data_set == "dev":
            model = "model-" + system + "-train.obj"
        else:
            model = "model-" + system + "-train+dev.obj"

        subprocess.call([
            "cort-predict",
            "-in", "/data/nlp/martscsn/thesis/data/input/" + data_set +
            ".auto",
            "-model", model,
            "-out", system + "-" + data_set + ".out",
            "-ante", system + "-" + data_set + ".antecedents",
            "-gold", "/data/nlp/martscsn/thesis/data/input/" + data_set +
            ".gold",
            "-extractor", get_extractor(data_set, system),
            "-perceptron", get_perceptron(system),
            "-clusterer", get_clusterer(system)])
