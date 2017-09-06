""" Extract instances and features from a corpus. """

import array
import multiprocessing
import sys

import mmh3
import numpy


__author__ = 'martscsn'


# for python 2 my_multiprocessing
def unwrap_extract_doc(arg, **kwarg):
    return InstanceExtractor._extract_doc(*arg, **kwarg)


class InstanceExtractor:
    """ Extract instances and their corresponding features from a corpus.

    Attributes:
        extract_substructures (function: CoNLLDocument ->
             list(list((Mention,Mention)))): Function for extracting the search
             space for a coreference resolution approach. The ith list in the
             nested list contains the search space for the ith substructure.
             The search space is represented as a nested list of mention pairs,
             which are candidate arcs in the graph to predict.
        mention_features (list(function: Mention -> str)): A list of features
            for mentions.
        pairwise_features (list(function: (Mention, Mention) -> str)): A list
            of features for mention pairs.
        cost_function (function: (Mention, Mention) -> int): A function
            assigning costs to mention pairs.
        labels (list(str)): A list of arc labels emplyoed by the approach.
            Defaults to the list containing only "+".
        convert_to_string_function (function): The function used to convert
            feature values to (unicode) strings. For Python 2 it is
            ``unicode``, for Python 3 it is ``string``.
    """
    def __init__(self,
                 extract_substructures,
                 mention_features,
                 pairwise_features,
                 cost_function,
                 labels=("+",)):
        """ Initialize instance and feature extraction.

        Args:
            extract_substructures (function: CoNLLDocument ->
                list(list((Mention,Mention)))): Function for extracting the
                search space for a coreference resolution approach. The ith
                list in the nested list contains the search space for the ith
                substructure. The search space is represented as a nested list
                of mention pairs, which are candidate arcs in the graph to
                predict.
            mention_features (list(function: Mention -> str)): A list of
                features for mentions.
            pairwise_features (list(function: (Mention, Mention) -> str)): A
                list of features for mention pairs.
            cost_function (function: (Mention, Mention) -> int): A function
                assigning costs to mention pairs.
            labels (list(str)): A list of arc labels emplyoed by the
                approach.
        """
        self.extract_substructures = extract_substructures
        self.mention_features = mention_features
        self.pairwise_features = pairwise_features
        self.cost_function = cost_function
        self.labels = labels

        if sys.version_info[0] == 2:
            self.convert_to_string_function = unicode
        else:
            self.convert_to_string_function = str

    def extract(self, corpus):
        """ Extract instances and features from a corpus.

        Args:
            corpus (Corpus): The corpus to extract instances and features from.

        Returns:
            A tuple which describes the extracted instances and their
            features. The individual components are:

            * substructures (list(list((Mention, Mention)))): The search space
                for the substructures, defined by a nested list. The ith list
                contains the search space for the ith substructure.
            * arc_information (dict((Mention, Mention),
                                    ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.
        """

        all_substructures = []
        arc_information = {}

        id_to_doc_mapping = {}
        for doc in corpus:
            id_to_doc_mapping[doc.identifier] = doc

        pool = multiprocessing.Pool(maxtasksperchild=1)

        if sys.version_info[0] == 2:
            results = pool.map(unwrap_extract_doc,
                               zip([self] * len(corpus.documents),
                                   corpus.documents))
        else:
            results = pool.map(self._extract_doc, corpus.documents)

        pool.close()
        pool.join()

        num_labels = len(self.labels)

        for result in results:
            (doc_identifier,
             anaphors,
             antecedents,
             nonnumeric_features,
             numeric_features,
             numeric_vals,
             costs,
             consistency,
             nonnumeric_feature_mapping,
             numeric_feature_mapping,
             substructures_mapping) = result

            doc = id_to_doc_mapping[doc_identifier]

            for i in range(0, len(substructures_mapping) - 1):
                struct = []
                begin = substructures_mapping[i]
                end = substructures_mapping[i + 1]

                for pair_index in range(begin, end):
                    arc = (doc.system_mentions[anaphors[pair_index]],
                           doc.system_mentions[antecedents[pair_index]])

                    struct.append(arc)

                    # find position of arc's features in document array
                    nonnumeric_features_start = nonnumeric_feature_mapping[
                        pair_index]
                    nonnumeric_features_end = nonnumeric_feature_mapping[
                        pair_index + 1]

                    numeric_features_start = numeric_feature_mapping[pair_index]
                    numeric_features_end = numeric_feature_mapping[
                        pair_index + 1]

                    arc_information[arc] = \
                        ((nonnumeric_features[
                          nonnumeric_features_start:nonnumeric_features_end
                          ],
                          numeric_features[
                          numeric_features_start:numeric_features_end
                          ],
                          numeric_vals[
                          numeric_features_start:numeric_features_end
                          ]),
                         costs[
                         num_labels * pair_index:num_labels * pair_index
                         + num_labels],
                         consistency[pair_index])

                all_substructures.append(struct)

        # in python 2, array.array does not support the buffer interface
        if sys.version_info[0] == 2:
            for arc in arc_information:
                feats, cost, cons = arc_information[arc]
                arc_information[arc] = (
                    (numpy.array(feats[0], dtype=numpy.uint32),
                     numpy.array(feats[1], dtype=numpy.uint32),
                     numpy.array(feats[2], dtype="float32")),
                    numpy.array(cost, dtype=float),
                    cons)

        return all_substructures, arc_information

    def _extract_doc(self, doc):
        cache = {}
        substructures = self.extract_substructures(doc)

        mentions_to_ids = {}

        for i, mention in enumerate(doc.system_mentions):
            mentions_to_ids[mention] = i

        anaphors = array.array('H')
        antecedents = array.array('H')
        costs = array.array('H')
        consistency = array.array('B')
        nonnumeric_feature_mapping = array.array('I')
        numeric_feature_mapping = array.array('I')
        substructures_mapping = array.array('I')
        nonnumeric_features = array.array('I')
        numeric_features = array.array('I')
        numeric_vals = array.array("f")

        nonnumeric_feature_mapping.append(0)
        numeric_feature_mapping.append(0)
        substructures_mapping.append(0)

        for struct in substructures:
            # skip empty
            if not struct:
                continue

            for arc in struct:
                # ids for anaphor and antecedent
                anaphors.append(mentions_to_ids[arc[0]])
                antecedents.append(mentions_to_ids[arc[1]])

                # cost for each label
                for label in self.labels:
                    costs.append(self.cost_function(arc, label))

                # is decision to make them coreferent consistent with gold?
                consistency.append(arc[0].decision_is_consistent(arc[1]))

                # features for the arc: stored in array which applies to whole
                # document
                (arc_nonnumeric_features, arc_numeric_features,
                 arc_numeric_vals) = self._extract_features(arc, cache)

                nonnumeric_features.extend(arc_nonnumeric_features)
                numeric_features.extend(arc_numeric_features)
                numeric_vals.extend(arc_numeric_vals)

                # auxiliary arrays that store the position of features for arcs
                # in the document array
                nonnumeric_feature_mapping.append(
                    nonnumeric_feature_mapping[-1] + len(
                        arc_nonnumeric_features))
                numeric_feature_mapping.append(
                    numeric_feature_mapping[-1] + len(arc_numeric_features))

            # store position of substructures in document array
            substructures_mapping.append(substructures_mapping[-1] +
                                         len(struct))

        return (doc.identifier,
                anaphors,
                antecedents,
                nonnumeric_features,
                numeric_features,
                numeric_vals,
                costs,
                consistency,
                nonnumeric_feature_mapping,
                numeric_feature_mapping,
                substructures_mapping)

    def _extract_features(self, arc, cache):
        anaphor, antecedent = arc
        inst_feats = []
        numeric_features = []

        numeric_types = {"float", "int"}

        if not antecedent.is_dummy():
            # mention features
            for mention in [anaphor, antecedent]:
                if mention not in cache:
                    cache[mention] = []
                    for feature in self.mention_features:
                        cache[mention].extend(feature(mention))

            ana_features = cache[anaphor]
            ante_features = cache[antecedent]

            # first: non-numeric features (categorial, boolean)
            inst_feats += ["ana_" + feat + "=" +
                           self.convert_to_string_function(val) for feat, val in
                           ana_features if type(val).__name__ not in
                           numeric_types]

            len_ana_features = len(inst_feats)

            inst_feats += ["ante_" + feat + "=" +
                           self.convert_to_string_function(val) for feat, val in
                           ante_features if type(val).__name__ not in
                           numeric_types]

            # concatenated features
            inst_feats += ["ana_" + ana_info[0] + "=" +
                           self.convert_to_string_function(ana_info[1]) +
                           "^ante_" + ante_info[0] + "=" +
                           self.convert_to_string_function(ante_info[1])
                           for ana_info, ante_info in
                           zip(ana_features, ante_features)]

            # pairwise features
            pairwise_features = []
            for feature in self.pairwise_features:
                pairwise_features.extend(feature(anaphor, antecedent))

            inst_feats += [feature + "=" +
                           self.convert_to_string_function(val) for feature, val
                           in pairwise_features
                           if val and type(val).__name__ not in numeric_types]

            # feature combinations
            fine_type_indices = {len_ana_features * i for i
                                 in [0, 1, 2]}

            inst_feats += [
                inst_feats[i] + "^" + word for i in fine_type_indices
                for j, word in enumerate(inst_feats)
                if j not in fine_type_indices
            ]

            # now numeric features
            ana_numeric = [("ana_" + feat, val) for feat, val
                           in ana_features
                           if type(val).__name__ in numeric_types]
            ante_numeric = [("ante_" + feat, val) for feat, val
                            in ante_features
                            if type(val).__name__ in numeric_types]
            pair_numeric = [(feat, val) for feat, val in pairwise_features
                            if type(val).__name__ in numeric_types]

            # feature combinations for numeric features
            for numeric_features in [ana_numeric, ante_numeric, pair_numeric]:
                numeric_features += [
                    (inst_feats[i] + "^" + numeric_features[j][0],
                     numeric_features[j][1]) for i in fine_type_indices
                    for j, numeric_feature in enumerate(numeric_features)
                ]

            numeric_features = ana_numeric + ante_numeric + pair_numeric

        # to hash
        all_nonnumeric_feats = array.array(
            'I', [mmh3.hash(word.encode("utf-8")) & 2 ** 24 - 1 for word
                  in inst_feats])
        all_numeric_feats = array.array(
            'I', [mmh3.hash(word.encode("utf-8")) & 2 ** 24 - 1 for word, _
                  in numeric_features])
        numeric_vals = array.array("f", [val for _, val in numeric_features])

        return all_nonnumeric_feats, all_numeric_feats, numeric_vals
