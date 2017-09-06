from collections import defaultdict

from cort.util import union_find

__author__ = 'martscsn'


class Clustering:
    """Represents a structured clustering of mentions in a document.

    This class is used by entity-based approaches to store the incremental
    construction of the document-wide coreference information.

    Attributes:
        mentions (list(Mention)): The mentions which are clustered.
        uf (UnionFind): A union-find data structure storing the clustering
            information.
        links (list((int, int))): A list of int tuples, where the tuple (i, j)
            denotes whether there is a link in the clustering from the ith
            mention in `mentions` to the the jth mention.
        outgoing_links (dict(int, int)): A mapping representing outgoing links.
            If `outgoing_links[i] = j`, then there is a link from the ith mention
            in `mentions` to the jth mention.
        mentions_to_clusters_mapping (dict(int, list(int)): A mapping of mention ids
            to clusters. if `mentions_to_clusters_mapping[i] = [j_1, ..., j_k]`, then
            the cluster of the ith mention in `mentions` consists of the
            j_1th, ..., j_kth mentions in `mentions`.
    """
    def __init__(self, mentions):
        self.mentions = mentions
        self.uf = union_find.UnionFind()
        for mention in self.mentions:
            self.uf.union(mention, mention)
        self.links = []
        self.outgoing_links = {}
        self.mentions_to_clusters_mapping = self._get_mention_to_clusters_mapping()

    def add_link(self, anaphor, antecedent):
        """ Add a link from `anaphor` to `antecedent` to the clustering.

        Args:
            anaphor (Mention): A mention.
            antecedent (Mention): Another mention.
        """
        self.links.append((anaphor, antecedent))
        self.outgoing_links[anaphor] = antecedent

        if not antecedent.is_dummy():
            self.uf.union(anaphor, antecedent)
            self.mentions_to_clusters_mapping = self._get_mention_to_clusters_mapping()

    def every_mention_has_antecedent(self):
        """ Return whether every mention in `self,mentions` has
         an antecedent.

         Returns;
            bool: Whether every mention in `self,mentions` has
                an antecedent.
        """
        return len(self.outgoing_links) == len(self.mentions) - 1

    def _get_mention_to_clusters_mapping(self):
        # for efficiency: implement this directly, not relying on similar method
        # (without ordering) in union_find.py
        repr_to_mentions = defaultdict(list)

        for m in sorted(self.uf, reverse=True):
            repr_to_mentions[self.uf[m]].append(m)

        mention_to_clusters = {}

        for mention in self.mentions:
            mention_to_clusters[mention] = \
                repr_to_mentions[self.uf[mention]]

        return mention_to_clusters

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.mentions == other.mentions
                    and self.links == other.links)
        else:
            return False
