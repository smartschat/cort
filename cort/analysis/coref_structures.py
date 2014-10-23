""" Structures for expressing coreference relations among a set of mentions. """


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
            (bool) True if the graphs have the same edges, False otherwise.
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
            (EntityGraph): The partition of this graph with respect to the
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
            (list((Mention, Mention)): Pairs of mentions describing edges that
                can be found in this graph, but not in the supplied graph.
        """
        difference = []

        for anaphor in self.edges:
            for antecedent in self.edges[anaphor]:
                if (anaphor not in entity_graph.edges or
                        antecedent not in entity_graph.edges[anaphor]):
                    difference.append((anaphor, antecedent))

        return difference
