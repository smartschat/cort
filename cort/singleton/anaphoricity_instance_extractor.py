""" Extract instances and features from a corpus. """

import array
import multiprocessing
import sys

import mmh3
import numpy


__author__ = 'martscsn, moosavi'

#reload(sys)
#sys.setdefaultencoding('utf-8')

# for python 2 multiprocessing
def unwrap_extract_doc(arg, **kwarg):
    return InstanceExtractor._extract_doc(*arg, **kwarg)


class InstanceExtractor:

    def __init__(self,
                 mention_features,
                 general_features,
                 labels=("+",)):

        self.mention_features = mention_features
        self.general_features = general_features
        self.labels = labels


    def get_consistency(self, mention):
        if mention.attributes['annotated_set_id'] is not None and not mention.attributes["first_in_gold_entity"]:
            return True
        return False    
    
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
                    arc = (doc.system_mentions[anaphors[pair_index]])

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
        mentions_to_ids = {}

        for i, mention in enumerate(doc.system_mentions):
            mentions_to_ids[mention] = i

        anaphors = array.array('H')
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
        for mention in doc.system_mentions[1:]:
            # ids for anaphor and antecedent
            anaphors.append(mentions_to_ids[mention])

            for label in self.labels:
                costs.append(0)
                
            consistency.append(self.get_consistency(mention))

            # features for the arc: stored in array which applies to whole
            # document
            #if self.target_type[0] == "NONE" or arc.attributes['type'] in self.target_type:
            (arc_nonnumeric_features, arc_numeric_features,
             arc_numeric_vals) = self._extract_features(mention, doc.system_mentions, cache)

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
                                         1)

        return (doc.identifier,
                anaphors,
                nonnumeric_features,
                numeric_features,
                numeric_vals,
                costs,
                consistency,
                nonnumeric_feature_mapping,
                numeric_feature_mapping,
                substructures_mapping)

    def _extract_features(self, arc, all_mentions, cache):
        anaphor = arc
        inst_feats = []
        numeric_features = []


        if anaphor not in cache:
            cache[anaphor] = [feature(anaphor) for feature
                                      in self.mention_features]

        mention_feats = cache[anaphor]

        inst_feats = [feat+"="+str(val) for feat, val in mention_feats]
        for feature in self.general_features:
            feat, val = feature(anaphor, all_mentions)
            inst_feats.append(feat+"="+str(val))
                
        inst_feats += [
            inst_feats[0] + "^" + word 
            for j, word in enumerate(inst_feats)
            if j !=0
        ]
        

        # to hash
        all_nonnumeric_feats = array.array(
            'I', [mmh3.hash(word.encode("utf-8")) & 2 ** 24 - 1 for word
                  in inst_feats])
        all_numeric_feats = array.array(
            'I', [mmh3.hash(word.encode("utf-8")) & 2 ** 24 - 1 for word, _
                  in numeric_features])
        numeric_vals = array.array("f", [val for _, val in numeric_features])
        return all_nonnumeric_feats, all_numeric_feats, numeric_vals
