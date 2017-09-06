""" Contains perceptron learning methods for structured prediction. """


from __future__ import division


import array
import sys

import numpy
cimport numpy
cimport cython
import mmh3

from cort.core.mentions import Mention

__author__ = 'smartschat'


cdef class Perceptron:
    cdef public double cost_scaling
    cdef public list cluster_features
    cdef public list dynamic_features

    cdef dict priors, weights, label_to_index

    """ Provide a latent structured perceptron.

    This implementation provides a latent structured perceptron with
    cost-augmented inference and parameter averaging for graphs encoding
    coreference decisions.

    Attributes:
        cost_scaling (int): The parameter for scaling the cost function during
            cost-augmented inference. Defaults to 1.
        priors (dict(str, float)): A mapping of graph labels to priors for
            these labels.
        weights (dict(str, array)): A mapping of labels to weight
            vectors. For each label ``l``, ``weights[l]`` contains weights
            for each feature seen during training (for representing the
            features we employ *feature hashing*). If the graphs employed are
            not labeled, ``l`` is set to "+".
        cluster_features (list(func)): A list of features operating on clusters
            of mentions.

    """

    def __init__(self,
                 cost_scaling=1,
                 cluster_features=None,
                 dynamic_features=None):
        """
        Initialize the perceptron.

        Args:
            cost_scaling (int): The parameter for scaling the cost function
                during cost-augmented inference. Defaults to 1.
            label (list(str)): A list of labels used in the graphs. If
                ``None``, defaults to ``["+"]``.
            priors (dict(str, float)): A mapping of graph labels to priors
                for these labels. If ``None`` defaults to an empty mapping.
            weights (dict(str, array)): A mapping of graph labels to
                arrays. For each label ``l``, ``weights[l]`` contains
                weights for each feature seen during training (for representing
                the features we employ *feature hashing*). If the graphs
                employed are not labeled, ``l`` is set to "+".
                If ``None`` defaults to a mapping of "+" to an array only
                containing 0s.
            cluster_features (list(func)): A list of features operating on clusters
                of mentions.
        """
        cdef double[:] weights_for_label
        cdef double[:] cached_weights_for_label

        self.counter = 0
        self.cost_scaling = cost_scaling

        labels = self.get_labels()

        self.label_to_index = {}
        for i, label in enumerate(labels):
            self.label_to_index[label] = i

        self.priors = {}
        self.cached_priors = {}
        for label in labels:
            self.priors[label] = 0.0
            self.cached_priors[label] = 0.0

        self.weights = {}
        self.cached_weights = {}

        try:
            for label in labels:
                weights_for_label = array.array("d", (0.0 for k in range(2**24)))
                cached_weights_for_label = array.array("d", (0.0 for k in range(2**24)))
                self.weights[label] = weights_for_label
                self.cached_weights[label] = cached_weights_for_label

        # in python 2, array.array does not support the buffer interface
        except TypeError:
            for label in labels:
                weights_for_label = numpy.zeros(2**24, dtype=float)
                cached_weights_for_label = numpy.zeros(2**24, dtype=float)
                self.weights[label] = weights_for_label
                self.cached_weights[label] = cached_weights_for_label

        if not cluster_features:
            cluster_features = []

        self.cluster_features = cluster_features
        self.dynamic_features = dynamic_features

        self.mode = None

    def get_cluster_features(self):
        return self.cluster_features

    def get_dynamic_features(self):
        return self.dynamic_features

    def fit_one_epoch(self, substructures, arc_information, indices):
        """Learn weights from data.

        The learned label priors and weights are stored in ``self.priors`` and
        ``self.weights``.

        Args:
            substructures (list(list((Mention, Mention)))): The search space
                for the substructures, defined by a nested list. The ith list
                contains the search space for the ith substructure.
            arc_information (dict((Mention, Mention),
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

        self.mode = "train"

        incorrect = 0
        decisions = 0

        converted_substructures = self._convert_substructures(substructures)

        for i in indices:
            decisions += 1

            (arcs,
             arcs_labels,
             arcs_scores,
             pred_additional_features,
             cons_arcs,
             cons_labels,
             cons_scores,
             cons_additional_features,
             is_consistent) = self.argmax(converted_substructures[i],
                                          arc_information)

            if not is_consistent:
                self.__update(cons_arcs,
                              arcs,
                              cons_labels,
                              arcs_labels,
                              arc_information,
                              cons_additional_features,
                              pred_additional_features)

                incorrect += 1

            self.counter += 1

        self.mode = None

        return incorrect, decisions

    def predict(self, substructures, arc_information):
        """
        Predict coreference information according to a learned model.

        Args:
            substructures (list(list((Mention, Mention)))): The search space
                for the substructures, defined by a nested list. The ith list
                contains the search space for the ith substructure.
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.
        Returns:
            Three nested lists describing the output. In particular, these
            lists are:

                - arcs (list(list(Mention, Mention))): The nested list of
                  predicted arcs. The ith list contains predictions for the
                  ith substructure.
                - labels (list(list(str))): Labels of the predicted arcs.
                - arcs (list(list(float))): Scores for the predicted arcs.
        """

        self.mode = "predict"

        arcs = []
        labels = []
        scores = []

        converted_substructures = self._convert_substructures(substructures)

        for i, substructure in enumerate(converted_substructures):
            (substructure_arcs,
             substructure_arcs_labels,
             substructure_arcs_scores,
             _,
             _,
             _,
             _,
             _,
             _) = self.argmax(substructure, arc_information)

            arcs.append(substructure_arcs)
            labels.append(substructure_arcs_labels),
            scores.append(substructure_arcs_scores)

        self.mode = None

        return arcs, labels, scores

    def argmax(self, substructure, arc_information):
        """ Decoder for coreference resolution.

        Compute highest-scoring substructure and highest-scoring constrained
        substructure consistent with the gold annotation. To implement
        coreference resolution approaches, inherit this class and implement
        this function.

        Args:
            substructure (list((Mention, Mention))): The list of mention pairs
                which define the search space for one substructure.
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.

        Returns:
            A 9-tuple describing the highest-scoring substructure and the
            highest-scoring substructure consistent with the gold information.
            The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the list of arcs
                   in the highest-scoring substructure,
                - **best_labels** (*list(str)*): the list of labels of the
                  arcs in the highest-scoring substructure,
                - **best_scores** (*list(float)*): the scores of the arcs in
                  the highest-scoring substructure,
                - **best_cluster_features** (*array(int)*): cluster features for
                  the best decision.
                - **best_cons_arcs** (*list((Mention, Mention))*): the list of
                  arcs in the highest-scoring constrained substructure
                  consistent with the gold information,
                - **best_cons_labels** (*list(str)*): the list of labels of the
                  arcs in the highest-scoring constrained substructure
                  consistent with the gold information,
                - **best_cons_scores** (*list(float)*): the scores of the arcs
                  in the highest-scoring constrained substructure consistent
                  with the gold information,
                - **best_const_cluster_features** (*array(int)*): cluster
                  features for the best decision constrained to be consistent
                  with the gold information.
                - **is_consistent** (*bool*): whether the highest-scoring
                  substructure is consistent with the gold information.
        """
        raise NotImplementedError()

    def __update(self, good_arcs, bad_arcs, good_labels, bad_labels,
                 arc_information, good_additional_features, bad_additional_features):

        if self.get_labels() == ["+"]:
            good_labels = ["+"]*(len(good_arcs) + 1)
            bad_labels = ["+"]*(len(bad_arcs) + 1)

        for arc, label, in zip(good_arcs, good_labels):
            nonnumeric_features, numeric_features, numeric_vals = arc_information[arc][0]

            self._update_cython(self.weights[label],
                           self.cached_weights[label],
                           nonnumeric_features,
                           numeric_features,
                           numeric_vals,
                           1.0,
                           1.0*self.counter)

            self.priors[label] += 1
            self.cached_priors[label] += self.counter



        for arc, label, in zip(bad_arcs, bad_labels):
            nonnumeric_features, numeric_features, numeric_vals = arc_information[arc][0]

            self._update_cython(self.weights[label],
                           self.cached_weights[label],
                           nonnumeric_features,
                           numeric_features,
                           numeric_vals,
                           -1.0,
                           -1.0*self.counter)

            self.priors[label] -= 1
            self.cached_priors[label] -= self.counter

        if good_additional_features:
            self._update_cython(self.weights[good_labels[0]],
                                self.cached_weights[good_labels[0]],
                                good_additional_features,
                                array.array("I"),
                                array.array("f"),
                                1.0,
                                1.0*self.counter)

        if bad_additional_features:
            self._update_cython(self.weights[bad_labels[0]],
                                self.cached_weights[bad_labels[0]],
                                bad_additional_features,
                                array.array("I"),
                                array.array("f"),
                                -1.0,
                                -1.0*self.counter)

    def get_labels(self):
        """ Get the graph labels employed by the current approach.

        In this case, the only graph label employed is '+'.

        Returns:
            list(str): A list of graph labels (contains only one graph label
                '+').

        """
        return ["+"]

    def get_coref_labels(self):
        """ Get the graph labels signalling coreference employed by the
        current approach.

        In this case, the only graph label signalling coreference is '+'.

        Returns:
            list(str): A list of graph labels signalling coreference (contains
                only one graph label '+').
        """
        return ["+"]

    def find_best_arcs(self, arcs, arc_information, label="+"):
        """ Find the highest-scoring arc and arc consistent with the gold
        information among a set of arcs.

        Args:
            arcs (list((Mention, Mention)): A list of mention pairs constituting
                arcs.
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.
            label (str): The label of the arcs. Defaults to "+".

        Returns:
            A 5-tuple describing the highest-scoring anaphor-antecedent
            decision, and the highest-scoring anaphor-antecedent decision
            consistent with the gold annotation. The tuple consists of:

                - **best** (*(Mention, Mention)*): the highest-scoring
                  anaphor-antecedent decision.
                - **max_val** (*float*): the score of the highest-scoring
                  anaphor-antecedent decision,
                - **best_cons** (*(Mention, Mention)*): the highest-scoring
                  anaphor-antecedent decision consistent with the gold
                  annotation.
                - **max_const** (*float*): the score of the highest-scoring
                  anaphor-antecedent decision consistent with the gold
                  annotation.
                - **is_consistent** (*bool*): whether the highest-scoring
                  anaphor-antecedent decision is consistent with the gold
                  information.
        """
        max_val = float("-inf")
        best = None

        max_cons = float("-inf")
        best_cons = None

        best_is_consistent = False

        for arc in arcs:
            features, costs, consistent = arc_information[arc]

            nonnumeric_features, numeric_features, numeric_vals = features

            score = self._cython_score_arc(self.priors[label],
                                      self.weights[label],
                                      self.cost_scaling,
                                      costs[self.label_to_index[label]],
                                      nonnumeric_features,
                                      numeric_features,
                                      numeric_vals)

            if score > max_val:
                best = arc
                max_val = score
                best_is_consistent = consistent

            if score > max_cons and consistent:
                best_cons = arc
                max_cons = score

        return best, max_val, best_cons, max_cons, best_is_consistent

    def score_arc(self, arc, arc_information, label="+", nocost=False):
        """ Score an arc according to priors, weights and costs.

        Args:
            arc ((Mention, Mention)): The pair of mentions constituting the arc.
            arc_information (dict((Mention, Mention),
                                  ((array, array, array), list(int), bool)):
                A mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features, the costs for
                the arc (for each label), and whether predicting the arc to be
                coreferent is consistent with the gold annotation). The features
                are divided in three arrays: the first array contains the non-
                numeric features, the second array the numeric features, and the
                third array the values for the numeric features. The features
                are represented as integers via feature hashing.
            label (str): The label of the arc. Defaults to "+".

        Returns:
            float: The sum of all weights for the features, plus the scaled
                costs for predicting the arc, plus the prior for the label.
        """

        features, costs, consistent = arc_information[arc]

        if nocost:
            cost = 0
        else:
            cost = costs[self.label_to_index[label]]

        nonnumeric_features, numeric_features, numeric_vals = features

        return self._cython_score_arc(
            self.priors[label],
            self.weights[label],
            self.cost_scaling,
            cost,
            nonnumeric_features,
            numeric_features,
            numeric_vals
        )

    def get_model(self):
        """ Get the (averaged) priors and weights of the current model.

        Returns:
            A tuple containing priors and weights. The tuple consists of:

            - **priors** (**dict(str, float)**): A mapping of graph labels to
                priors for these labels.
            - **weights** (**dict(str, array)**): A mapping of graph labels to
                arrays. For each label ``l``, ``weights[l]`` contains
                weights for each feature seen during training (for representing
                the features we employ *feature hashing*). If the graphs
                employed are not labeled, ``l`` is set to "+".
        """
        model_priors = {}
        model_weights = {}

        cdef double[:] model_weights_for_label

        for label in self.priors:
            model_priors[label] = self.priors[label]
            model_weights_for_label = array.array("d", self.weights[label])
            model_weights[label] = model_weights_for_label
            model_priors[label] -= (1/self.counter)*self.cached_priors[label]
            self._average_weights(model_weights[label],
                                  self.cached_weights[label],
                                  1.0*self.counter)
            model_weights[label] = array.array("d", model_weights[label])

        return model_priors, model_weights

    def load_model(self, model):
        """ Get the (averaged) priors and weights of the current model.

        Returns:
            A tuple containing priors and weights. The tuple consists of:

            - **priors** (**dict(str, float)**): A mapping of graph labels to
                priors for these labels.
            - **weights** (**dict(str, array)**): A mapping of graph labels to
                arrays. For each label ``l``, ``weights[l]`` contains
                weights for each feature seen during training (for representing
                the features we employ *feature hashing*). If the graphs
                employed are not labeled, ``l`` is set to "+".
        """
        cdef double[:] weights_for_label

        priors, weights = model

        self.priors = priors

        for label in self.priors:
            weights_for_label = weights[label]
            self.weights[label] = weights_for_label

    def get_weights(self, features, vals=None, label="+"):
        if vals:
            return self._cython_get_weights_with_vals(self.weights[label],
                                                      features, vals)
        else:
            return self._cython_get_weights(self.weights[label], features)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _cython_get_weights_with_vals(self,
                                    double[:] weights,
                                    numpy.uint32_t[:] features,
                                    float[:] vals) nogil:

        cdef double score = 0.0
        cdef int index = 0

        for index in range(features.shape[0]):
            score += weights[features[index]]*vals[index]

        return score

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _cython_get_weights(self,
                                    double[:] weights,
                                    numpy.uint32_t[:] features) nogil:

        cdef double score = 0.0
        cdef int index = 0

        for index in range(features.shape[0]):
            score += weights[features[index]]

        return score

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _cython_score_arc(self,
                                  double prior,
                                  double[:] weights,
                                  double cost_scaling,
                                  double costs,
                                  numpy.uint32_t[:] nonnumeric_features,
                                  numpy.uint32_t[:] numeric_features,
                                  float[:] numeric_vals):

        cdef double score = 0.0
        cdef int index = 0

        score += prior
        score += cost_scaling * costs

        for index in range(nonnumeric_features.shape[0]):
            score += weights[nonnumeric_features[index]]

        for index in range(numeric_features.shape[0]):
            score += weights[numeric_features[index]]*numeric_vals[index]

        return score

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _update_cython(self,
                             double[:] weights,
                             double[:] cached_weights,
                             numpy.uint32_t[:] nonnumeric_features,
                             numpy.uint32_t[:] numeric_features,
                             float[:] numeric_vals,
                             double update_val_for_weights,
                             double update_val_for_cached_weights):
        cdef int index

        for index in range(nonnumeric_features.shape[0]):
            weights[nonnumeric_features[index]] += update_val_for_weights
            cached_weights[nonnumeric_features[index]] += \
                update_val_for_cached_weights

        for index in range(numeric_features.shape[0]):
            weights[numeric_features[index]] += \
                update_val_for_weights*numeric_vals[index]
            cached_weights[numeric_features[index]] += \
                update_val_for_cached_weights*numeric_vals[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _average_weights(self, double[:] weights, double[:] cached_weights,
                          double counter):
        cdef int index

        for index in range(weights.shape[0]):
            weights[index] -= cached_weights[index]/counter

    def _convert_substructures(self, substructures):
        return substructures

    def _get_rescored(self,
                      ana,
                      candidates,
                      substructure,
                      cluster_feats,
                      dynamic_feats):

        max_antecedent = -1
        max_score = float("-inf")
        max_cluster_features = None
        max_dynamic_features = None

        dummy = Mention.dummy_from_document(ana.document)

        ana_cluster = [m for m in
                       substructure.mentions_to_clusters_mapping[ana]]

        for score, ante in candidates:
            ante_cluster = [m for m in
                            substructure.mentions_to_clusters_mapping[ante]]

            cluster_features = (array.array("I"), array.array("I"), array.array("I"))

            cluster_weight = 0

            if cluster_feats and ante_cluster != [dummy] and ante_cluster != []:
                # generate cluster features...
                cluster_features = featurize(ana_cluster, ante_cluster,
                                             substructure,
                                             cluster_feats)
                cluster_nonnumeric_features, cluster_numeric_features, \
                cluster_numeric_vals = cluster_features

                # ...and compute weight
                cluster_weight = self.get_weights(cluster_nonnumeric_features)
                cluster_weight += self.get_weights(cluster_numeric_features,
                                                   vals=cluster_numeric_vals)

            dynamic_features = (array.array("I"), array.array("I"), array.array("I"))

            dynamic_weight = 0

            # generate dynamic pairwise features...
            if dynamic_feats and not ante.is_dummy():
                dynamic_features = featurize(ana, ante, substructure,
                                             dynamic_feats)
                dynamic_nonnumeric_features, dynamic_numeric_features, \
                dynamic_numeric_vals = dynamic_features

                # ...and compute weight
                dynamic_weight += self.get_weights(dynamic_nonnumeric_features)
                dynamic_weight += self.get_weights(dynamic_numeric_features,
                                                   vals=dynamic_numeric_vals)

            rescored = score + cluster_weight + dynamic_weight

            if rescored > max_score or (rescored == max_score
                                        and ana.attributes["id"] - ante.attributes["id"] < ana.attributes["id"] - max_antecedent.attributes["id"]):
                max_antecedent = ante
                max_score = rescored
                max_cluster_features = cluster_features[0] + cluster_features[1]
                max_dynamic_features = dynamic_features[0] + dynamic_features[1]

        # return highest scoring
        return (
            max_score,
            (ana,
             max_antecedent),
            max_cluster_features + max_dynamic_features
        )

def featurize(object_a, object_b, substructure, features):
    numeric_types = {"float", "int"}

    non_numeric_feats = []
    numeric_feats = []
    numeric_vals = []

    if sys.version_info[0] == 2:
        convert_to_string_function = unicode
    else:
        convert_to_string_function = str

    features_and_vals = []

    for feature in features:
        features_and_vals.extend(feature(object_a, object_b, substructure))

    for feature, val in features_and_vals:
        if val is None:
            continue
        if type(val).__name__ in numeric_types:
            numeric_feats.append(feature)
            numeric_vals.append(val)
        else:
            non_numeric_feats.append(feature + "=" +
                                     convert_to_string_function(val))

    # to hash
    all_nonnumeric_feats = array.array(
        'I', [mmh3.hash(word.encode("utf-8")) & 2 ** 24 - 1 for word
              in non_numeric_feats])
    all_numeric_feats = array.array(
        'I', [mmh3.hash(word.encode("utf-8")) & 2 ** 24 - 1 for word
              in numeric_feats])
    all_numeric_vals = array.array("f", numeric_vals)

    return all_nonnumeric_feats, all_numeric_feats, all_numeric_vals