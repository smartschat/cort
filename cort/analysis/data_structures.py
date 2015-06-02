""" Structures useful for error analysis. """


from collections import defaultdict


__author__ = 'smartschat'


class EntityGraph:
    """ Represent coreference relation between mentions in a graph.

    An entity graph consists of edges from anaphors to all coreferent mentions
    earlier in the text.

    More formally, an entity graph is a graph with the following properties:

        - the nodes are mentions
        - all edges point "backwards": if m appears later in the text than n,
          then there cannot exist an edge from n to m
        - there is an edge between to mentions if and only if they are
          coreferent

    Attributes:
        edges (dict(Mention, list(Mention))): A mapping from mentions to all
            mentions which have an incoming edge from that mention.

    """

    def __init__(self, edges):
        """ Initialize an entity graph from edges.

        Args:
            edges (dict(Mention, list(Mention))): A mapping from mentions to all
                mentions which have an incoming edge from that mention.
        """
        self.edges = edges

    def __eq__(self, other):
        """ Compare graphs for equality.

        Two graphs are equal if they have the same edges.

        Args:
            other (EntityGraph): An entity graph

        Returns:
            True if the graphs have the same edges, False otherwise.
        """
        if isinstance(other, self.__class__):
            return self.edges == other.edges
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(frozenset([(key, tuple(value))
                    for key, value in self.edges.items()]))

    def __repr__(self):
        return repr(self.edges)

    def __str__(self):
        return str(self.edges)

    @staticmethod
    def from_mentions(mentions, id_attribute):
        """ Construct a set of entity graphs from mentions.

        In particular, build a complete graph for every subset of mentions that
        have the same id_attribute.

        Args:
            mentions (list(Mention)): A list of mentions.
            id_attribute (str): The mention attribute which should be used to
                determine coreference. Possible values are "annotated_set_id"
                and "set_id".

        Returns:
            list(EntityGraph): A list of entity graphs. Two mentions are in one
            entity graph if and only if they have the same id_attribute.
        """
        id_to_mentions = defaultdict(list)

        for mention in mentions:
            if mention.attributes[id_attribute] is None:
                continue
            else:
                id_to_mentions[
                    mention.attributes[id_attribute]].append(mention)

        graphs = [EntityGraph.__create_complete(sorted(mention_list))
                      for mention_list in id_to_mentions.values()
                      if len(mention_list) > 1]

        return graphs

    @staticmethod
    def __create_complete(mentions):
        edges = {}
        for i in range(1, len(mentions)):
            edges[mentions[i]] = sorted(mentions[0:i], reverse=True)

        return EntityGraph(edges)

    def partition(self, entity_graphs):
        """ Partition the entity graph with respect to a set of entity graphs.

        The partitioned graph is a subgraph of this entity graph. An edge is
        retained if it is found in some graph in the supplied set of entity
        graphs.

        Args:
            entity_graphs (list(EntityGraph)): A list of entity graphs.

        Returns:
            EntityGraph: The partition of this graph with respect to the
            supplied graphs.
        """
        edges = {}
        for anaphor in self.edges:
            for antecedent in self.edges[anaphor]:
                if EntityGraph.__in_some_entity_graph(anaphor, antecedent,
                                                      entity_graphs):
                    if anaphor not in edges:
                        edges[anaphor] = list()
                    edges[anaphor].append(antecedent)

        return EntityGraph(edges)

    @staticmethod
    def __in_some_entity_graph(anaphor, antecedent, entity_graphs):
        for entity_graph in entity_graphs:
            if EntityGraph.__in_entity_graph(anaphor, antecedent, entity_graph):
                return True

    @staticmethod
    def __in_entity_graph(anaphor, antecedent, entity_graph):
        return anaphor in entity_graph.edges and \
            antecedent in entity_graph.edges[anaphor]

    def difference(self, entity_graph):
        """ Get all pairs of mention that are in this graph, but not in the
        supplied entity graph.

        Args:
            entity_graph (EntityGraph): An entity graph

        Returns:
            list((Mention, Mention): Pairs of mentions describing edges that
            can be found in this graph, but not in the supplied graph.
        """
        difference = []

        for anaphor in self.edges:
            for antecedent in self.edges[anaphor]:
                if (anaphor not in entity_graph.edges or
                        antecedent not in entity_graph.edges[anaphor]):
                    difference.append((anaphor, antecedent))

        return difference


class EnhancedSet:
    """ Manage, filter and categorize sets.

    For example, this class can manage coreference resolution errors or
    antecedent decisions.

    Attributes:
        data (set): A set.
    """

    def __init__(self, data=None):
        """ Initialize an enhanced set.

        Args:
            data: Any collection of data, defaults to None.
        """
        if data:
            self.data = set(data)
        else:
            self.data = set()

    def __iter__(self):
        return self.data.__iter__()

    def __contains__(self, datum):
        return datum in self.data

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        if isinstance(other, EnhancedSet):
            return self.data == other.data
        else:
            return False

    def __hash__(self):
        return hash(self.data)

    def __repr__(self):
        return sorted(self.data).__repr__()

    def __str__(self):
        return sorted(self.data).__str__()

    def filter(self, function):
        """ Return a new EnhancedSet by filtering with respect to a function.

        All members ``x`` for which ``if function(x)`` holds are retained, the
        remaining are discarded. Typically, the function will be a boolean
        function, and therefore correspond to a predicate. For example, if the
        members are mention pairs corresponding to coreference resolution
        errors,
        ``lambda x: x[0].attributes["type"] == "NAM"``,
        will retain all errors with a proper name anaphor.

        Args:
            function: A function mapping members to values.

        Returns:
            EnhancedSet: An EnhancedSet filtered by the function.
        """
        filtered = set()

        for datum in self.data:
            if function(datum):
                filtered.add(datum)

        return EnhancedSet(filtered)

    def categorize(self, categorizer,
                   corpora=None,
                   reference=None):
        """ Categorize members with respect to a categorizer.

        Compute a category for each member, and then return a mapping of
        categories to EnhancedSets of members which have the respective
        category.

        For example, for coreference resolution errors, one typical
        categorizer is the mention type mapping of anaphor and
        antecedent, which is computed by the function
        ``lambda x: (x[0].attributes["type"], x[1].attributes["type"])``.

        Args:
            categorizer (function): A function mapping members to categories.

        Returns
            StructuredCoreferenceAnalysis: A StructuredCoreferenceAnalysis
                containing the categorized items. All items in one category
                can be accessed like in a dict, for example
                ``categorized[("NOM", "NOM")]``, if ``output`` is the
                StructuredCoreferenceAnalysis.
        """
        categorized = {}

        for datum in self.data:
            category = categorizer(datum)

            if category not in categorized:
                categorized[category] = EnhancedSet()

            categorized[category].data.add(datum)

        return StructuredCoreferenceAnalysis(categorized, corpora, reference)

    def intersection(self, other):
        """ Return the intersection of this EnhancedSet and another EnhancedSet.

        Args:
            other (EnhancedSet): An EnhancedSet.

        Returns
            EnhancedSet: All members which are in this EnhancedSet and in the
                other EnhancedSet.
        """
        return EnhancedSet(self.data.intersection(other.data))

    def difference(self, other):
        """ Return the difference of this EnhancedSet and another EnhancedSet.

        Args:
            other (EnhancedSet): An EnhancedSet.

        Returns
            EnhancedSet: All members which are in this EnhancedSet but not in
                the other EnhancedSet.
        """
        return EnhancedSet(self.data.difference(other.data))


class StructuredCoreferenceAnalysis:
    """ Manage EnhancedSets containing coreference information, for example
    errors or antecedent decisions.

    The included EnhancedSets can  be categorized and subcategorized to an
    arbitrary level. Access to these subcategorized sets works like in a dict,
    for example as in
    ``enhanced_mapping["pair"]["recall_errors"]["all"]``

    Attributes:
        mapping (dict(object, StructuredCoreferenceAnalysis)): A mapping of
            categories ``to StructuredCoreferenceAnalysis`` objects..
    """
    def __init__(self, input_mapping, corpora, reference):
        """ Initialize an StructuredCoreferenceAnalysis with a dict.

        Recursively transform the dict into a StructuredCoreferenceAnalysis.

        Args:
            input_mapping (dict): A nested dict. Leaves of the dict should be
                EnhancedSets.
            corpora (dict(str, Corpus)):  A mapping of corpus names to the
                respective Corpus objects. Stores information which system
                corpora are included in this StructuredCoreferenceAnalysis.
            reference (Corpus): The reference corpus for entries in this
                StructuredCoreferenceAnalysis.
        """
        self.corpora = corpora
        self.reference = reference
        mapping = {}

        for key, val in input_mapping.items():
            if isinstance(val, EnhancedSet):
                mapping[key] = val
            else:
                mapping[key] = StructuredCoreferenceAnalysis(
                    input_mapping[key], corpora, reference)

        self.mapping = mapping

    def __iter__(self):
        """ Iterate over all errors (recursively) contained in this
        StructuredCoreferenceAnalysis.
        """
        for key, val in self.mapping.items():
            if isinstance(val, StructuredCoreferenceAnalysis):
                for inner_val in val.__iter__():
                    yield inner_val
            else:
                for datum in val:
                    yield datum

    def __len__(self):
        """ Return the number of all errors contained in this
        StructuredCoreferenceAnalysis.

        Returns:
            int: The number of all errors contained in this
                StructuredCoreferenceAnalysis.
        """
        total_len = 0

        for key, val in self.mapping.items():
            total_len += len(val)

        return total_len

    def __eq__(self, other):
        if isinstance(other, StructuredCoreferenceAnalysis):
            return self.mapping == other.mapping
        else:
            return False

    def __hash__(self):
        return hash(self.mapping)

    def __repr__(self):
        return repr(self.mapping)

    def __str__(self):
        return str(self.mapping)

    def __getitem__(self, key):
        return self.mapping[key]

    def keys(self):
        return self.mapping.keys()

    def items(self):
        return self.mapping.items()

    def categorize(self, categorizer):
        """ Categorize EnhancedSets in this analysis with respect to a
        categorizer.

        Compute a category for each member in the EnhancedSets in this
        analysis. Add the respective categories at the highest nesting level.

        For example, for coreference resolution errors, one typical
        categorizer is the mention type mapping of anaphor and
        antecedent, which is computed by the function
        ``lambda x: (x[0].attributes["type"], x[1].attributes["type"])``.

        If the highest level before could be accessed as
        ``self["pair"]["recall_errors"]["all"]``,
        now the highest level can be accessed with
        ``self["pair"]["recall_errors"]["all"][("NOM", "NOM")]``,
        ``self["pair"]["recall_errors"]["all"][("PRO", "NAM")]``, ...

        Args:
            categorizer (function): A function mapping members to categories.

        Returns
            StructuredCoreferenceAnalysis: The analysis enriched with the new
                categories.
        """
        categorized_mapping = {}

        self._construct_helper(categorized_mapping, self.mapping, [],
                               categorizer, "categorize")

        return StructuredCoreferenceAnalysis(categorized_mapping,
                                             self.corpora,
                                             self.reference)

    def filter(self, function):
        """ Filter contained EnhancedSets with respect to a function.

        All members ``x`` in the EnhancedSets for which ``if function(x)``
        holds are retained, the remaining are discarded. Typically, the
        function will be a boolean  function, and therefore correspond to a
        predicate. For example, if the members are mention pairs
        corresponding to coreference resolution errors,
        ``lambda x: x[0].attributes["type"] == "NAM"``,
        will retain all errors with a proper name anaphor.

        Args:
            function: A function mapping members to values.

        Returns:
            StructuredCoreferenceAnalysis: A StructuredCoreferenceAnalysis
                where the EnhancedSets are filtered with respect to the
                function.
        """
        filtered_mapping = {}

        self._construct_helper(filtered_mapping, self.mapping, [],
                               function, "filter")

        return StructuredCoreferenceAnalysis(filtered_mapping,
                                             self.corpora,
                                             self.reference)

    def update(self, other):
        """ Update this StructuredCoreferenceAnalysis with another
        StructuredCoreferenceAnalysis.

        Entries from the other StructuredCoreferenceAnalysis have priority.

        Args:
            other (StructuredCoreferenceAnalysis): Another
                StructuredCoreferenceAnalysis.
        """
        self._construct_helper(self.mapping, other, [], lambda x: True,
                               "filter")

    def _construct_helper(self, constructed_mapping, mapping, position,
                          function, construct_type):
        curr_mapping = mapping
        for key in position:
            curr_mapping = curr_mapping[key]

        for key in curr_mapping.keys():
            val = curr_mapping[key]
            if isinstance(val, EnhancedSet):
                if construct_type == "categorize":
                    constructed_mapping[key] = val.categorize(function,
                                                              self.corpora,
                                                              self.reference)
                elif construct_type == "filter":
                    constructed_mapping[key] = val.filter(function)
            else:
                if key not in constructed_mapping:
                    new_corpora = self.corpora

                    if isinstance(mapping, StructuredCoreferenceAnalysis):
                        new_corpora.update(mapping.corpora)

                    constructed_mapping[key] = \
                        StructuredCoreferenceAnalysis({}, new_corpora,
                                                      self.reference)

                self._construct_helper(constructed_mapping[key].mapping,
                                       mapping,
                                       position + [key],
                                       function,
                                       construct_type)

    def visualize(self, corpus_name, error=None):
        """ Visualize errors contained in this StructuredCoreferenceAnalysis.

        In particular, visualize all recall and precision errors for the
        system ``corpus_name``. Errors can be categorized, but no nested
        categorization is supported for visualization. That is, visualize
        all errors that can be found under

        ``self[corpus_name]["recall_errors"]["all"]`` and
        ``self[corpus_name]["precision"]["all"]``,

        or, if the errors are categorized, under

        ``self[corpus_name]["recall_errors"]["all"][category]`` for each
        category ``category``.

        The errors are written to an HTML file, which is copied to
        ``temp/output/error_analysis.html``. This file is opened automatically.

        Args:
            corpus_name (str): The name (description attribute) of the corpus
                which shall be visualized.
            error ((Mention, Mention), optional): An error to highlight.
                Defaults to None.

        """
        from cort.analysis import visualization
        visualization.Visualizer(self, corpus_name, error).run()
