__author__ = 'sebastian'


import codecs


from cort.analysis import error_extractors
from cort.analysis import plotting
from cort.analysis import spanning_tree_algorithms
from cort.core import corpora

# read in corpora
reference = corpora.Corpus.from_file("reference", codecs.open("dev.gold", "r",
                                                              "utf-8"))
pair = corpora.Corpus.from_file("pair", codecs.open("pair-dev.out", "r", "utf-8"))
tree = corpora.Corpus.from_file("tree", codecs.open("tree-dev.out", "r", "utf-8"))

# optional -- not needed when you only want to compute recall errors
pair.read_antecedents(open('pair-dev.antecedents'))
tree.read_antecedents(open('tree-dev.antecedents'))

# define error extractor
extractor = error_extractors.ErrorExtractor(
    reference,
    spanning_tree_algorithms.recall_accessibility,
    spanning_tree_algorithms.precision_system_output
)

# extract errors
extractor.add_system(pair)
extractor.add_system(tree)

errors = extractor.get_errors()

errors.update(pair.get_antecedent_decisions())
errors.update(tree.get_antecedent_decisions())

# categorize by mention type of anaphor
by_type = errors.categorize(
    lambda err: err[0].attributes["type"]
)


# visualize
by_type.visualize("pair")

# filter by distance
by_type_filtered = by_type.filter(
    lambda err: err[0].attributes["sentence_id"] - err[1].attributes[
        "sentence_id"] <= 3
)

# plot
pair_errs = by_type_filtered["pair"]["recall_errors"]["all"]
tree_errs = by_type_filtered["tree"]["recall_errors"]["all"]

plotting.plot(
    [("pair", [(cat, len(errs)) for cat, errs in pair_errs.items()]),
     ("tree", [(cat, len(errs)) for cat, errs in tree_errs.items()])],
    "Recall Errors",
    "Type of anaphor",
    "Number of Errors")

# more advanced features

# is anaphor a gold mention?
all_gold = set()
for doc in reference:
    for mention in doc.annotated_mentions:
        all_gold.add(mention)


def is_anaphor_gold(mention):
    if mention in all_gold:
        return "is_gold"
    else:
        return "is_not_gold"

is_ana_gold = by_type.categorize(lambda err: is_anaphor_gold(err[0]))

# head statistics for NOM errors
from collections import Counter

for system in ["pair", "tree"]:
    nom_rec_errs = by_type[system]["recall_errors"]["all"]["NOM"]
    all_heads = [" ".join(err[0].attributes["head"]).lower() for err in nom_rec_errs]
    most_common = Counter(all_heads).most_common(10)
    print(system, most_common)

# common errors:
common = {
    "common": {
        "recall_errors": {},
        "precision_errors": {}
    }
}

common["common"]["recall_errors"]["all"] = errors["pair"]["recall_errors"][
    "all"].intersection(errors["tree"]["recall_errors"]["all"])

common["common"]["precision_errors"]["all"] = errors["pair"]["precision_errors"][
    "all"].intersection(errors["tree"]["precision_errors"]["all"])

from cort.analysis import data_structures
common = data_structures.StructuredCoreferenceAnalysis(
    common, errors.reference, errors.corpora
)

# plot decisions
decs = by_type_filtered["pair"]["decisions"]["all"]
prec_errs = by_type_filtered["pair"]["precision_errors"]["all"]

plotting.plot(
    [("decisions", [(cat, len(errs)) for cat, errs in decs.items()]),
     ("errors", [(cat, len(errs)) for cat, errs in prec_errs.items()])],
    "Decisions and Errors",
    "Type of anaphor",
    "Number")