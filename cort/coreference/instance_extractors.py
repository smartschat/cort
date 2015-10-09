""" Extract instances and features from a corpus. """


import array
import multiprocessing
import sys


import mmh3
import numpy


__author__ = 'martscsn'


# for python 2 multiprocessing
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
            * arc_information (dict((Mention, Mention), (array, int, bool)): A
                mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features (represented as
                an int array via feature hashing), the costs for the arc (for
                each label), and whether predicting the arc to be coreferent is
                consistent with the gold annotation).
        """

        all_substructures = []
        arc_information = {}

        id_to_doc_mapping = {}
        for doc in corpus:
            id_to_doc_mapping[doc.identifier] = doc

        pool = multiprocessing.Pool(maxtasksperchild=1)

        if sys.version_info[0] == 2:
            results = pool.map(unwrap_extract_doc,
                               zip([self]*len(corpus.documents),
                                   corpus.documents))
        else:
            results = pool.map(self._extract_doc, corpus.documents)

        pool.close()
        pool.join()

        num_labels = len(self.labels)

        for result in results:
            doc_identifier, anaphors, antecedents, features, costs, \
                consistency, feature_mapping, substructures_mapping = result

            doc = id_to_doc_mapping[doc_identifier]

            for i in range(0, len(substructures_mapping)-1):
                struct = []
                begin = substructures_mapping[i]
                end = substructures_mapping[i+1]

                for pair_index in range(begin, end):
                    arc = (doc.system_mentions[anaphors[pair_index]],
                           doc.system_mentions[antecedents[pair_index]])

                    struct.append(arc)

                    features_start = feature_mapping[pair_index]
                    features_end = feature_mapping[pair_index+1]

                    arc_information[arc] = \
                        (features[features_start:features_end],
                         costs[num_labels*pair_index:num_labels*pair_index + num_labels],
                         consistency[pair_index])

                all_substructures.append(struct)

        # in python 2, array.array does not support the buffer interface
        if sys.version_info[0] == 2:
            for arc in arc_information:
                feats, cost, cons = arc_information[arc]
                arc_information[arc] = (numpy.array(feats, dtype=numpy.uint32),
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
        feature_mapping = array.array('I')
        substructures_mapping = array.array('I')
        features = array.array('I')

        feature_mapping.append(0)
        substructures_mapping.append(0)

        for struct in substructures:
            # skip empty
            if not struct:
                continue

            for arc in struct:
                anaphors.append(mentions_to_ids[arc[0]])
                antecedents.append(mentions_to_ids[arc[1]])
                for label in self.labels:
                    costs.append(self.cost_function(arc, label))
                consistency.append(arc[0].decision_is_consistent(arc[1]))

                arc_features = self._extract_features(arc, cache)
                features.extend(arc_features)
                feature_mapping.append(feature_mapping[-1] + len(arc_features))

            substructures_mapping.append(substructures_mapping[-1] +
                                         len(struct))

        return (doc.identifier,
                anaphors,
                antecedents,
                features,
                costs,
                consistency,
                feature_mapping,
                substructures_mapping)

    def _extract_features(self, arc, cache):
        anaphor, antecedent = arc
        inst_feats = []

        if not antecedent.is_dummy():
            # mention features
            for mention in [anaphor, antecedent]:
                if mention not in cache:
                    cache[mention] = [feature(mention) for feature
                                      in self.mention_features]

            ana_features = cache[anaphor]
            ante_features = cache[antecedent]

            inst_feats += ["ana_" + feat for feat in ana_features]
            inst_feats += ["ante_" + feat for feat in ante_features]

            # concatenated features
            inst_feats += ["ana_" + feat_ana + "^ante_" + feat_ante
                           for feat_ana, feat_ante in
                           zip(ana_features, ante_features)]

            # pairwise features
            inst_feats += [feature(anaphor, antecedent) for feature
                           in self.pairwise_features
                           if feature(anaphor, antecedent)]

            # feature combinations
            fine_type_indices = {len(self.mention_features)*i for i
                                 in [0, 1, 2]}

            inst_feats += [
                inst_feats[i] + "^" + word for i in fine_type_indices
                for j, word in enumerate(inst_feats)
                if j not in fine_type_indices
            ]

        # to hash
        all_feats = array.array('I', [mmh3.hash(word.encode("utf-8")) & 2**24-1
                                      for word in inst_feats])

        return all_feats