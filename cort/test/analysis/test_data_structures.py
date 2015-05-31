from collections import defaultdict
import unittest

from cort.analysis import data_structures
from cort.core import documents
from cort.core import mentions
from cort.core import spans


__author__ = 'smartschat'


class TestCorefStructures(unittest.TestCase):
    def setUp(self):
        self.complicated_mention_example = """#begin	document	(test2);	part	000
test2	0	0	This    NN   (NP*	-   -   -   -   -   (0)
test2	0	1	is  NN	*   -   -   -   -   -   -
test2	0	2	just    NN   *	-   -   -   -   -   -
test2	0	3	a   NN   *	-   -   -   -   -   (0|(1)
test2	0	4	test    NN   *	-   -   -   -   -   0)
test2	0	5	.   NN   *)	-   -   -   -   -   -

test2	0	0	It  NN   (NP*	-   -   -   -   -   (1)|(0
test2	0	1	shows   NN   *	-   -   -   -   -   -
test2	0	2	that    NN   *	-   -   -   -   -   (2)
test2	0	3	the NN   *	-   -   -   -   -   (2|(3
test2	0	4	scorer  NN   *	-   -   -   -   -   2)|0)
test2	0	5	works   NN   *	-   -   -   -   -   3)
test2	0	6	.   NN   *)	-   -   -   -   -   -

#end	document"""

        self.complicated_mention_document = documents.CoNLLDocument(
            self.complicated_mention_example)

    def test_entity_graph_from_mentions(self):
        annotated_mentions = \
            self.complicated_mention_document.annotated_mentions

        first_graph = data_structures.EntityGraph({
            annotated_mentions[4]: [annotated_mentions[2],
                                    annotated_mentions[0]],
            annotated_mentions[2]: [annotated_mentions[0]]
        })

        second_graph = data_structures.EntityGraph({
            annotated_mentions[3]: [annotated_mentions[1]]
        })

        third_graph = data_structures.EntityGraph({
            annotated_mentions[6]: [annotated_mentions[5]]
        })

        self.assertEqual(
            [first_graph, second_graph, third_graph],
            data_structures.EntityGraph.from_mentions(annotated_mentions,
                                                       "annotated_set_id"))

    def test_entity_graph_partition(self):
        annotated_mentions = \
            self.complicated_mention_document.annotated_mentions

        graph = data_structures.EntityGraph({
            annotated_mentions[4]: [annotated_mentions[2],
                                    annotated_mentions[0]],
            annotated_mentions[2]: [annotated_mentions[0]]
        })

        system_output = [
            mentions.Mention(
                self.complicated_mention_document,
                spans.Span(0, 0),
                {"set_id": 0}),
            mentions.Mention(
                self.complicated_mention_document,
                spans.Span(2, 3),
                {"set_id": 1}),
            mentions.Mention(
                self.complicated_mention_document,
                spans.Span(6, 10),
                {"set_id": 0}),
            mentions.Mention(
                self.complicated_mention_document,
                spans.Span(5, 5),
                {"set_id": 0})
        ]

        expected_edges = defaultdict(list)
        expected_edges[annotated_mentions[4]].append(annotated_mentions[0])
        expected = data_structures.EntityGraph(expected_edges)

        self.assertEqual(expected,
                         graph.partition(
                             data_structures.EntityGraph.from_mentions(
                                 system_output, "set_id")))


if __name__ == '__main__':
    unittest.main()