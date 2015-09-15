""" Extract errors made by systems w.r.t. a reference corpus. """


from cort.analysis import data_structures


__author__ = 'smartschat'


class ErrorExtractor:
    """ Extract, manage and store recall and precision errors.

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
        recall_spanning_tree_algorithm (function): A function mapping an
            entity graph and one its partitions to a list of mentions pairs,
            which represent a spanning tree of the entity graph. This
            function is used to compute recall errors.
        precision_spanning_tree_algorithm (function): Same as above, but for
            precision errors.
        errors (dict): A mapping of error descriptions to sets containing the
            respective errors.
    """
    def __init__(self,
                 reference_corpus,
                 recall_spanning_tree_algorithm,
                 precision_spanning_tree_algorithm,
                 ):
        """ Initialize the error analysis.

        Args:
            reference_corpus (Corpus): The reference corpus with the gold
                information concerning the coreference relation.
            recall_spanning_tree_algorithm (function): A function mapping an
                entity graph and one its partitions to a list of mentions pairs,
                which represent a spanning tree of the entity graph. This
                function is used to compute recall errors.
            precision_spanning_tree_algorithm (function): Same as above, but for
                precision errors.
        """

        self.reference_corpus = reference_corpus
        self.recall_spanning_tree_algorithm = recall_spanning_tree_algorithm
        self.precision_spanning_tree_algorithm = \
            precision_spanning_tree_algorithm
        self.errors = {}
        self.corpora = {}

    def add_system(self, system_corpus, which_mentions="annotated"):
        """ Add a system to the error analysis.

        Error extraction for recall errors works as follows:

        Go through each document. For each reference entity e in the document,
        construct an entity graph g_e for e and compute a partition of g_e by
        the system entity graphs. Then compute a spanning tree t_e of g_e and
        take every edge in t_e that does not appear in the partition as an
        error.

        For computing precision errors, switch the roles of reference and system
        entities.

        Also extracts all pairwise decisions (if available).

        Args:
            system_corpus (Corpus): A corpus obtained from system output.
            which_mentions (str): Either "annotated" or "extracted",
                defaults to "annotated". Specifies from which mentions in
                the system corpus coreference information should be
                obtained, either annotated mentions or system mentions.
        """
        if which_mentions not in ["annotated", "extracted"]:
            raise ValueError("which_mentions must be"
                             "either 'annotated' or 'extracted'.")

        recall_errors, precision_errors = self.__compute_errors(system_corpus,
                                                                which_mentions)

        self.errors[system_corpus.description] = {
            "recall_errors": {},
            "precision_errors": {},
            "decisions": {}
        }

        self.errors[system_corpus.description]["recall_errors"]["all"] = \
            recall_errors
        self.errors[
            system_corpus.description]["precision_errors"]["all"] = \
            precision_errors
        self.errors[
            system_corpus.description]["decisions"]["all"] = \
            system_corpus.get_antecedent_decisions()[
            system_corpus.description]["decisions"]["all"]

        self.corpora[system_corpus.description] = system_corpus

    def get_errors(self):
        """ Get errors for all systems managed by this ErrorAnalysis.

        The errors are stored via an ``StructuredCoreferenceAnalysis`
        which can be accessed like a dict.

        If a corpus with the description
        ``ranking``was added via ``self.add_system``,
        ``self.errors["ranking"]["recall_errors"]["all"]``is an ``EnhancedSet``
        containing all recall errors of the system. Errors of other systems
        and precision errors can be accessed analogously.

        Returns:
            StructuredCoreferenceAnalysis: A StructuredCoreferenceAnalysis
                containing the errors.
        """
        return data_structures.StructuredCoreferenceAnalysis(
            self.errors, corpora=self.corpora,
            reference=self.reference_corpus)

    def __compute_errors(self, system_corpus, which_mentions):
        gold_graphs = [data_structures.EntityGraph.from_mentions(
            doc.annotated_mentions, "annotated_set_id")
            for doc in self.reference_corpus.documents]

        if which_mentions == 'annotated':
            system_graphs = [data_structures.EntityGraph.from_mentions(
                doc.annotated_mentions, "annotated_set_id")
                for doc in system_corpus.documents]
        else:
            system_graphs = [data_structures.EntityGraph.from_mentions(
                doc.system_mentions, "set_id")
                for doc in system_corpus.documents]

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

        return (data_structures.EnhancedSet(recall_errors),
                data_structures.EnhancedSet(precision_errors))

    @staticmethod
    def __compute_errors_for_doc(base_graphs,
                                 partitioning_graphs,
                                 spanning_tree_algorithm):
        errors = []

        for graph in base_graphs:
            errors.extend(
                ErrorExtractor.__compute_errors_for_graph(
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
