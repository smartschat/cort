import unittest

from cort.core.documents import CoNLLDocument
from cort.coreference import data_structures


__author__ = 'martscsn'


class TestDataStructures(unittest.TestCase):
    def setUp(self):
        self.complicated_mention_example = """#begin	document	(test2);	part	000
        test2	0	0	This    NN   (NP*	-   -   -   -   -   (0)
        test2	0	1	is  NN	*   -   -   -   -   -   -
        test2	0	2	just    NN   *	-   -   -   -   -   -
        test2	0	3	a   NN   *	-   -   -   -   -   (0|(1)
        test2	0	4	test    NN   *	-   -   -   -   -   0)
        test2	0	5	.   NN   *)	-   -   -   -   -   -

        test2	0	0	It  NN   (NP*	-   -   -   -   -   (1)|(4
        test2	0	1	shows   NN   *	-   -   -   -   -   -
        test2	0	2	that    NN   *	-   -   -   -   -   (2)
        test2	0	3	the NN   *	-   -   -   -   -   (2|(3
        test2	0	4	scorer  NN   *	-   -   -   -   -   2)|4)
        test2	0	5	works   NN   *	-   -   -   -   -   3)
        test2	0	6	.   NN   *)	-   -   -   -   -   -

        #end	document"""

        self.complicated_mention_document = CoNLLDocument(self.complicated_mention_example)

    def test_clustering(self):
        mentions = self.complicated_mention_document.annotated_mentions

        my_clustering = data_structures.Clustering(mentions)

        my_clustering.add_link(mentions[3], mentions[1])

        expected_mentions_to_clusters_mapping = {
            mentions[0]: [mentions[0]],
            mentions[1]: [mentions[3], mentions[1]],
            mentions[2]: [mentions[2]],
            mentions[3]: [mentions[3], mentions[1]],
            mentions[4]: [mentions[4]],
            mentions[5]: [mentions[5]],
            mentions[6]: [mentions[6]],
            mentions[7]: [mentions[7]]

        }

        self.assertEqual(
            expected_mentions_to_clusters_mapping,
            my_clustering.mentions_to_clusters_mapping
        )

