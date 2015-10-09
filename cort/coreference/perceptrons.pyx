""" Contains perceptron learning methods for structured prediction. """


from __future__ import division

import logging
import array

import numpy

cimport numpy
cimport cython


__author__ = 'smartschat'


cdef class Perceptron:
    cdef int n_iter, random_seed
    cdef double cost_scaling
    cdef dict priors, weights, label_to_index
    """ Provide a latent structured perceptron.

    This implementation provides a latent structured perceptron with
    cost-augmented inference and parameter averaging for graphs encoding
    coreference decisions.

    Attributes:
        n_iter (int): The number of epochs for training. Defaults to 5.
        seed (int): The random seed for shuffling the data. Defaults to 23.
        cost_scaling (int): The parameter for scaling the cost function during
            cost-augmented inference. Defaults to 1.
        priors (dict(str, float)): A mapping of graph labels to priors for
            these labels.
        weights (dict(str, array)): A mapping of labels to weight
              vectors. For each label ``l``, ``weights[l]`` contains weights
              for each feature seen during training (for representing the
              features we employ *feature hashing*). If the graphs employed are
              not labeled, ``l`` is set to "+".
        label_to_index (dict(str, int)): A mapping of labels to the index of the
              label in the list of labels.

    """

    def __init__(self,
                 n_iter=5,
                 seed=23,
                 cost_scaling=1,
                 priors=None,
                 weights=None):
        """
        Initialize the perceptron.

        Args:
            n_iter (int): The number of epochs for training. Defaults to 5.
            seed (int): The random seed for shuffling the data. Defaults to 23.
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
        """
        cdef double[:] weights_for_label

        self.n_iter = n_iter
        self.random_seed = seed
        self.cost_scaling = cost_scaling

        labels = self.get_labels()

        self.label_to_index = {}
        for i, label in enumerate(labels):
            self.label_to_index[label] = i

        if not priors:
            self.priors = {}
            for label in labels:
                self.priors[label] = 0.0
        else:
            self.priors = priors

        self.weights = {}

        try:
            if weights is None:
                for label in labels:
                    weights_for_label = array.array("d",
                                                    (0.0 for k in range(2**24)))
                    self.weights[label] = weights_for_label
            else:
                for label in weights:
                    weights_for_label = weights[label]
                    self.weights[label] = weights_for_label
        # in python 2, array.array does not support the buffer interface
        except TypeError:
            if weights is None:
                for label in labels:
                    weights_for_label = numpy.zeros(2**24, dtype=float)
                    self.weights[label] = weights_for_label
            else:
                for label in weights:
                    weights_for_label = numpy.array(weights[label])
                    self.weights[label] = weights_for_label

    def fit(self, substructures, arc_information):
        """Learn weights from data.

        Besides returning the learned model, also
        set the corresponding attributes ``self.priors``and ``self.weights``.

        Args:
            substructures (list(list((Mention, Mention)))): The search space
                for the substructures, defined by a nested list. The ith list
                contains the search space for the ith substructure.
            arc_information (dict((Mention, Mention), (array, array, bool)): A
                mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features (represented as
                an int array via feature hashing), the costs for the arc (for
                each label, order as in self.get_labels()), and whether
                predicting the arc to be coreferent is consistent with the gold
                annotation).
        Returns:
            A tuple describing the learned model, consisting of

                - **priors** (*dict(str, float)*): A mapping of graph labels to
                  priors for these labels.
                - **weights** (*dict(str, array)*): A mapping of graph
                  labels to arrays. For each label ``l``, ``weights[l]``
                  contains weights for each feature seen during training
                  (for representing the features we employ *feature hashing*).
                  If the graphs employed are not labeled, ``l`` is set to "+".
        """

        cdef double[:] cached_weights_for_label

        indices = list(range(0, len(substructures)))
        numpy.random.seed(self.random_seed)

        cached_priors = {}
        cached_weights = {}

        for label in self.priors:
            cached_priors[label] = 0.0

            try:
                cached_weights_for_label = array.array("d",
                                                    (0.0 for k in range(2**24)))
            except TypeError:
                cached_weights_for_label = numpy.zeros(2**24, dtype=float)

            cached_weights[label] = cached_weights_for_label

        counter = 0

        for epoch in range(1, self.n_iter+1):
            logging.info("Started epoch " + str(epoch))
            numpy.random.shuffle(indices)

            incorrect = 0

            for i in indices:
                substructure = substructures[i]

                (arcs,
                 arcs_labels,
                 arcs_scores,
                 cons_arcs,
                 cons_labels,
                 cons_scores,
                 is_consistent) = self.argmax(substructure,
                                              arc_information)

                if not is_consistent:
                    self.__update(cons_arcs,
                                  arcs,
                                  cons_labels,
                                  arcs_labels,
                                  arc_information,
                                  counter,
                                  cached_priors,
                                  cached_weights)
                    incorrect += 1

                counter += 1

            logging.info("Finished epoch " + str(epoch))
            logging.info("\tIncorrect predictions: " + str(incorrect) + "/" +
                         str(len(indices)))

        # averaging
        for label in self.priors:
            self.priors[label] -= (1/counter)*cached_priors[label]
            self._average_weights(self.weights[label], cached_weights[label],
                                  1.0*counter)

    def predict(self, substructures, arc_information):
        """
        Predict coreference information according to a learned model.

        Args:
            substructures (list(list((Mention, Mention)))): The search space
                for the substructures, defined by a nested list. The ith list
                contains the search space for the ith substructure.
            arc_information (dict((Mention, Mention), (array, array, bool)): A
                mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features (represented as
                an int array via feature hashing), the costs for the arc (for
                each label, order as in self.get_labels()), and whether
                predicting the arc to be coreferent is consistent with the gold
                annotation).
        Returns:
            Three nested lists describing the output. In particular, these
            lists are:

                - arcs (list(list(Mention, Mention))): The nested list of
                  predicted arcs. The ith list contains predictions for the
                  ith substructure.
                - labels (list(list(str))): Labels of the predicted arcs.
                - arcs (list(list(float))): Scores for the predicted arcs.
        """
        arcs = []
        labels = []
        scores = []

        for substructure in substructures:
            (substructure_arcs, substructure_arcs_labels,
             substructure_arcs_scores, _, _, _, _) = self.argmax(
                substructure, arc_information)

            arcs.append(substructure_arcs)
            labels.append(substructure_arcs_labels)
            scores.append(substructure_arcs_scores)

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
            arc_information (dict((Mention, Mention), (array, array, bool)): A
                mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features (represented as
                an int array via feature hashing), the costs for the arc (for
                each label, order as in self.get_labels()), and whether
                predicting the arc to be coreferent is consistent with the gold
                annotation).

        Returns:
            A 6-tuple describing the highest-scoring substructure and the
            highest-scoring substructure consistent with the gold information.
            The tuple consists of:

                - **best_arcs** (*list((Mention, Mention))*): the list of arcs
                   in the highest-scoring substructure,
                - **best_labels** (*list(str)*): the list of labels of the
                  arcs in the highest-scoring substructure,
                - **best_scores** (*list(float)*): the scores of the arcs in
                  the highest-scoring substructure,
                - **best_cons_arcs** (*list((Mention, Mention))*): the list of
                  arcs in the highest-scoring constrained substructure
                  consistent with the gold information,
                - **best_cons_labels** (*list(str)*): the list of labels of the
                  arcs in the highest-scoring constrained substructure
                  consistent with the gold information,
                - **best_cons_scores** (*list(float)*): the scores of the arcs
                  in the highest-scoring constrained substructure consistent
                  with the gold information,
                - **is_consistent** (*bool*): whether the highest-scoring
                  substructure is consistent with the gold information.
        """
        raise NotImplementedError()

    def __update(self, good_arcs, bad_arcs, good_labels, bad_labels,
                 arc_information, counter, cached_priors, cached_weights):

        if good_labels or bad_labels:
            for arc, label in zip(good_arcs, good_labels):
                self._update_cython(self.weights[label],
                               cached_weights[label],
                               arc_information[arc][0],
                               1.0,
                               1.0*counter)

                self.priors[label] += 1
                cached_priors[label] += counter

            for arc, label in zip(bad_arcs, bad_labels):
                self._update_cython(self.weights[label],
                               cached_weights[label],
                               arc_information[arc][0],
                               -1.0,
                               -1.0*counter)

                self.priors[label] -= 1
                cached_priors[label] -= counter
        else:
            for arc in good_arcs:
                self._update_cython(self.weights["+"],
                               cached_weights["+"],
                               arc_information[arc][0],
                               1.0,
                               1.0*counter)

            for arc in bad_arcs:
                self._update_cython(self.weights["+"],
                               cached_weights["+"],
                               arc_information[arc][0],
                               -1.0,
                               -1.0*counter)

    def get_labels(self):
        return ["+"]

    def get_coref_labels(self):
        return ["+"]

    def find_best_arcs(self, arcs, arc_information, label="+"):
        max_val = float("-inf")
        best = None

        max_cons = float("-inf")
        best_cons = None

        best_is_consistent = False

        for arc in arcs:
            features, costs, consistent = arc_information[arc]

            score = self._cython_score_arc(self.priors[label],
                                      self.weights[label],
                                      self.cost_scaling,
                                      costs[self.label_to_index[label]],
                                      features)

            if score > max_val:
                best = arc
                max_val = score
                best_is_consistent = consistent

            if score > max_cons and consistent:
                best_cons = arc
                max_cons = score

        return best, max_val, best_cons, max_cons, best_is_consistent

    def score_arc(self, arc, arc_information, label="+"):
        """ Score an arc according to priors, weights and costs.

        Args:
            arc ((Mention, Mention)): The pair of mentions constituting the arc.
            arc_information (dict((Mention, Mention), (array, array, bool)): A
                mapping of arcs (= mention pairs) to information about these
                arcs. The information consists of the features (represented as
                an int array via feature hashing), the costs for the arc (for
                each label, order as in self.get_labels()), and whether
                predicting the arc to be coreferent is consistent with the gold
                annotation).
            label (str): The label of the arc. Defaults to "+".

        Returns:
            float: The sum of all weights for the features, plus the scaled
                costs for predicting the arc, plus the prior for the label.
        """

        features, costs, consistent = arc_information[arc]

        return self._cython_score_arc(
            self.priors[label],
            self.weights[label],
            self.cost_scaling,
            costs[self.label_to_index[label]],
            features
        )

    def get_model(self):
        array_weights = {}

        for label in self.weights:
            array_weights[label] = array.array("d", self.weights[label])

        return self.priors, array_weights


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _cython_score_arc(self,
                                  double prior,
                                  double[:] weights,
                                  double cost_scaling,
                                  double costs,
                                  numpy.uint32_t[:] features) nogil:

        cdef double score = 0.0
        cdef int index = 0

        score += prior
        score += cost_scaling * costs

        for index in range(features.shape[0]):
            score += weights[features[index]]

        return score

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _update_cython(self,
                             double[:] weights,
                             double[:] cached_weights,
                             numpy.uint32_t[:] features,
                             double update_val_for_weights,
                             double update_val_for_cached_weights):
        cdef int index

        for index in range(features.shape[0]):
            weights[features[index]] += update_val_for_weights
            cached_weights[features[index]] += update_val_for_cached_weights

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _average_weights(self, double[:] weights, double[:] cached_weights,
                          double counter):
        cdef int index

        for index in range(weights.shape[0]):
            weights[index] -= cached_weights[index]/counter