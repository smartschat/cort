""" Represent errors made by a system w.r.t. a reference corpus. """


from cort.analysis import coref_structures


__author__ = 'smartschat'


class ErrorSet:
    """ Manage, filter and categorize coreference resolution errors.

    Attributes:
        errors (set): A set of errors.
    """

    def __init__(self, errors=None):
        """ Initialize an error set with errors.

        Args:
            errors: Any collection of errors, defaults to None.
        """
        if errors:
            self.errors = set(errors)
        else:
            self.errors = set()

    def __iter__(self):
        return self.errors.__iter__()

    def __len__(self):
        return len(self.errors)

    def __eq__(self, other):
        if isinstance(other, ErrorSet):
            return self.errors == other.errors
        else:
            return False

    def __repr__(self):
        return sorted(self.errors).__repr__()

    def __str__(self):
        return sorted(self.errors).__str__()

    def filter(self, function):
        """ Return a new ErrorSet by filtering with respect to a function.

        All errors for which
            if function(error)
        holds are retained, the remaining are discarded. Typically, the function
        will be a boolean function, and therefore correspond to a predicate,
        like
            lambda x: x[0].attributes["type"] == "NAM",
        which will retain all errors with a proper name anaphor.

        Args:
            function: A function mapping errors to values.

        Returns:
            (ErrorSet): An ErrorSet filtered by the function.
        """
        filtered = set()

        for error in self.errors:
            if function(error):
                filtered.add(error)

        return ErrorSet(filtered)

    def categorize(self, categorizer):
        """ Categorize errors in this set with respect to a categorizer.

        Compute a category for each error, and then return a mapping of
        categories to errors which have the respective category.

        Args:
            categorizer (function): A function mapping errors to categories.

        Returns
            (dict(object, ErrorSet)): A mapping of categories to errors which
                have the respective category.
        """
        categorized = {}

        for error in self.errors:
            category = categorizer(error)

            if category not in categorized:
                categorized[category] = ErrorSet()

            categorized[category].errors.add(error)

        return categorized

    def intersection(self, other):
        """ Return the intersection of this ErrorSet and another ErrorSet.

        Args:
            other (ErrorSet): An ErrorSet.

        Returns
            (ErrorSet): All errors which are in this ErrorSet and in the other
                ErrorSet.
        """
        return ErrorSet(self.errors.intersection(other.errors))


class ErrorAnalysis:
    """ Extract and store recall and precision errors made by a system.

    Error extraction for recall errors works as follows:

    Go through each document. For each reference entity e in the document,
    construct an entity graph g_e for e and compute a partition of g_e by the
    system entity graphs. Then compute a spanning tree t_e of g_e and take
    every edge in t_e that does not appear in the partition as an error.

    For computing precision errors, switch the roles of reference and system
    entities.

    Attributes:
        reference_corpus (Corpus): The reference corpus with the gold
            information concerning the coreference relation.
        system_corpus (Corpus): The corpus obtained from system output.
        recall_spanning_tree_algorithm (function): A function mapping an
            entity graph and one its partitions to a list of mentions pairs,
            which represent a spanning tree of the entity graph. This
            function is used to compute recall errors.
        precision_spanning_tree_algorithm (function): Same as above, but for
            precision errors.
        which_mentions: (str): Either "annotated" or "extracted", defaults
            to "annotated". Specifies from which mentions in the system
            corpus coreference information should be obtained, either
            annotated mentions or system mentions.
    """
    def __init__(self,
                 reference_corpus,
                 system_corpus,
                 recall_spanning_tree_algorithm,
                 precision_spanning_tree_algorithm,
                 which_mentions="annotated"):
        """ Store the corpora and compute errors.

        Args:
            reference_corpus (Corpus): The reference corpus with the gold
                information concerning the coreference relation.
            system_corpus (Corpus): The corpus obtained from system output.
            error_set_creator (ErrorSetFromCorpusCreator): An error set
                creator, which will be used to extract errors.
            which_mentions (str): either "annotated" or "extracted",
                defaults to "annotated". Specifies whether system
                coreference information should be extracted from annotated
                mentions or system mentions.
        """
        if which_mentions not in ["annotated", "extracted"]:
            raise ValueError("which_mentions must be"
                             "either 'annotated' or 'extracted'.")

        self.reference_corpus = reference_corpus
        self.system_corpus = system_corpus
        self.recall_spanning_tree_algorithm = recall_spanning_tree_algorithm
        self.precision_spanning_tree_algorithm = \
            precision_spanning_tree_algorithm
        self.which_mentions = which_mentions
        self.recall_errors, self.precision_errors = self.__compute_errors()

    def __compute_errors(self):
        gold_graphs = [coref_structures.EntityGraph.from_mentions(
            doc.annotated_mentions, "annotated_set_id")
            for doc in self.reference_corpus.documents]

        if self.which_mentions == 'annotated':
            system_graphs = [coref_structures.EntityGraph.from_mentions(
                doc.annotated_mentions, "annotated_set_id")
                for doc in self.system_corpus.documents]
        else:
            system_graphs = [coref_structures.EntityGraph.from_mentions(
                doc.system_mentions, "set_id")
                for doc in self.system_corpus.documents]

        recall_errors = []
        precision_errors = []

        for doc_gold_graphs, doc_system_graphs in zip(gold_graphs,
                                                      system_graphs):
            recall_errors.extend(
                self.__compute_errors_for_doc(
                    doc_gold_graphs,
                    doc_system_graphs,
                    self.recall_spanning_tree_algorithm))
            precision_errors.extend(
                self.__compute_errors_for_doc(
                    doc_system_graphs,
                    doc_gold_graphs,
                    self.precision_spanning_tree_algorithm))

        return (ErrorSet(recall_errors),
                ErrorSet(precision_errors))

    @staticmethod
    def __compute_errors_for_doc(base_graphs,
                                 partitioning_graphs,
                                 spanning_tree_algorithm):
        errors = []

        for graph in base_graphs:
            errors.extend(
                ErrorAnalysis.__compute_errors_for_graph(
                    graph, partitioning_graphs, spanning_tree_algorithm))

        return errors

    @staticmethod
    def __compute_errors_for_graph(graph,
                                   partitioning_graphs,
                                   spanning_tree_algorithm):
        partitioned_graph = graph.partition(partitioning_graphs)
        spanning_tree = spanning_tree_algorithm(graph, partitioned_graph)
        extra_pairs = [
            (anaphor, antecedent) for anaphor, antecedent in spanning_tree
            if anaphor not in partitioned_graph.edges or
            antecedent not in partitioned_graph.edges[anaphor]
        ]

        return [(anaphor, antecedent) for anaphor, antecedent in sorted(
            extra_pairs)]
