import unittest
from collections import defaultdict

from cort.analysis import data_structures
from cort.analysis import spanning_tree_algorithms
from cort.core import mentions
from cort.core import spans


__author__ = 'smartschat'


class TestSpanningTreeAlgorithms(unittest.TestCase):
    def setUp(self):
        self.gold_first_cluster = [
            mentions.Mention(
                None,
                spans.Span(0, 0),
                {"tokens": ["a", "rainbow"],
                 "type": "NOM",
                 "tokens_as_lowercase_string": "a rainbow",
                 "head_as_lowercase_string": "rainbow",
                 "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(1, 1),
                {"tokens": ["US"],
                 "type": "NAM",
                 "tokens_as_lowercase_string": "us",
                 "head_as_lowercase_string": "us",
                 "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(2, 3),
                {"tokens": ["angry", "salesman"],
                 "type": "PRO",
                 "tokens_as_lowercase_string": "angry salesman",
                 "head_as_lowercase_string": "salesman",
                 "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(4, 5),
                {"tokens": ["the", "rainbow"],
                 "type": "NAM",
                 "tokens_as_lowercase_string": "the rainbow",
                 "head_as_lowercase_string": "rainbow",
                 "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(5, 6),
                {"tokens": ["and", "far"],
                 "type": "NOM",
                 "tokens_as_lowercase_string": "and far",
                 "head_as_lowercase_string": "and",
                 "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(7, 7),
                {"tokens": ["neypmd"],
                 "type": "NOM",
                 "tokens_as_lowercase_string": "neypmd",
                 "head_as_lowercase_string": "neypmd",
                 "annotated_set_id": 0}),
        ]

        self.gold_second_cluster = [
            mentions.Mention(
                None,
                spans.Span(7, 8),
                {"type": "NOM", "annotated_set_id": 1}),

            mentions.Mention(
                None,
                spans.Span(9, 9),
                {"type": "NAM", "annotated_set_id": 1}),

            mentions.Mention(
                None,
                spans.Span(10, 10),
                {"type": "PRO", "annotated_set_id": 1}),
        ]

        self.system1_mentions = [
            mentions.Mention(None, spans.Span(0, 0), {"set_id": 0}),
            mentions.Mention(None, spans.Span(2, 3), {"set_id": 0}),
            mentions.Mention(None, spans.Span(4, 5), {"set_id": 2}),
            mentions.Mention(None, spans.Span(5, 6), {"set_id": 2}),
            mentions.Mention(None, spans.Span(3, 4), {"set_id": 1}),
            mentions.Mention(None, spans.Span(7, 8), {"set_id": 1}),
        ]

        self.system2_cluster = [
            mentions.Mention(
                None,
                spans.Span(0, 0),
                {"tokens": ["a"], "set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(2, 3),
                {"tokens": ["angry", "salesman"], "set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(7, 8),
                {"tokens": ["snafu", "foo"], "set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(9, 9),
                {"tokens": ["bar"], "set_id": 0}),
        ]
        self.system2_cluster[1].attributes["antecedent"] = \
            [self.system2_cluster[0]]
        self.system2_cluster[2].attributes["antecedent"] = \
            [self.system2_cluster[0]]
        self.system2_cluster[3].attributes["antecedent"] = \
            [self.system2_cluster[2]]

        self.maxDiff = None

    def test_recall_closest(self):
        gold_graph = data_structures.EntityGraph.from_mentions(
            self.gold_first_cluster, "annotated_set_id", {})[0]

        spanning_tree_edges = [
            (self.gold_first_cluster[1], self.gold_first_cluster[0]),
            (self.gold_first_cluster[2], self.gold_first_cluster[0]),
            (self.gold_first_cluster[3], self.gold_first_cluster[2]),
            (self.gold_first_cluster[4], self.gold_first_cluster[3]),
            (self.gold_first_cluster[5], self.gold_first_cluster[4])
        ]

        self.assertEqual(
            spanning_tree_edges,
            spanning_tree_algorithms.recall_closest(
                gold_graph,
                gold_graph.partition(
                    data_structures.EntityGraph.from_mentions(
                        self.system1_mentions, "set_id", {}))))

    def test_recall_accessibility(self):
        gold_graph = data_structures.EntityGraph.from_mentions(
            self.gold_first_cluster, "annotated_set_id", {})[0]

        spanning_tree_edges = [
            (self.gold_first_cluster[1], self.gold_first_cluster[0]),
            (self.gold_first_cluster[2], self.gold_first_cluster[0]),
            (self.gold_first_cluster[3], self.gold_first_cluster[1]),
            (self.gold_first_cluster[4], self.gold_first_cluster[3]),
            (self.gold_first_cluster[5], self.gold_first_cluster[3])
        ]

        self.assertEqual(
            spanning_tree_edges,
            spanning_tree_algorithms.recall_accessibility(
                gold_graph,
                gold_graph.partition(
                    data_structures.EntityGraph.from_mentions(
                        self.system1_mentions, "set_id", {}))))

    def test_recall_improved_accessibility(self):
        gold_graph = data_structures.EntityGraph.from_mentions(
            self.gold_first_cluster, "annotated_set_id", {})[0]

        spanning_tree_edges = [
            (self.gold_first_cluster[1], self.gold_first_cluster[0]),
            (self.gold_first_cluster[2], self.gold_first_cluster[0]),
            (self.gold_first_cluster[3], self.gold_first_cluster[0]),
            (self.gold_first_cluster[4], self.gold_first_cluster[3]),
            (self.gold_first_cluster[5], self.gold_first_cluster[3])
        ]

        self.assertEqual(
            spanning_tree_edges,
            spanning_tree_algorithms.recall_improved_accessibility(
                gold_graph,
                gold_graph.partition(
                    data_structures.EntityGraph.from_mentions(
                        self.system1_mentions, "set_id", {}))))

    def test_precision_system_output(self):
        gold_graph = data_structures.EntityGraph.from_mentions(
            self.system2_cluster, "set_id", {})[0]

        spanning_tree_edges = [
            (self.system2_cluster[1], self.system2_cluster[0]),
            (self.system2_cluster[2], self.system2_cluster[0]),
            (self.system2_cluster[3], self.system2_cluster[2])
        ]

        self.assertEqual(
            spanning_tree_edges,
            spanning_tree_algorithms.precision_system_output(
                gold_graph,
                gold_graph.partition(
                    data_structures.EntityGraph.from_mentions(
                        self.gold_first_cluster, "annotated_set_id", {}))))

    def test_precision_scores(self):
        scores = defaultdict(float)

        scores.update({
            (self.system2_cluster[1], self.system2_cluster[0]): -1,
            (self.system2_cluster[2], self.system2_cluster[0]): -2,
            (self.system2_cluster[2], self.system2_cluster[1]): 2,
            (self.system2_cluster[3], self.system2_cluster[0]): 3.5,
            (self.system2_cluster[3], self.system2_cluster[1]): 3.5,
            (self.system2_cluster[3], self.system2_cluster[2]): 1.2,
        })

        system_graph = data_structures.EntityGraph.from_mentions(
            self.system2_cluster, "set_id", scores)[0]

        spanning_tree_edges = [
            (self.system2_cluster[1], self.system2_cluster[0]),
            (self.system2_cluster[2], self.system2_cluster[1]),
            (self.system2_cluster[3], self.system2_cluster[1])
        ]

        self.assertEqual(
            spanning_tree_edges,
            spanning_tree_algorithms.precision_scores(
                system_graph,
                system_graph.partition(
                    data_structures.EntityGraph.from_mentions(
                        self.gold_first_cluster, "annotated_set_id", scores))))

    def test_recall_scores(self):
        scores = defaultdict(float)

        scores.update({
            (self.gold_first_cluster[1], self.gold_first_cluster[0]): -1,
            (self.gold_first_cluster[2], self.gold_first_cluster[0]): 0,
            (self.gold_first_cluster[2], self.gold_first_cluster[1]): -1,
            (self.gold_first_cluster[3], self.gold_first_cluster[0]): 0,
            (self.gold_first_cluster[3], self.gold_first_cluster[1]): 0.5,
            (self.gold_first_cluster[3], self.gold_first_cluster[2]): 0.25,
            (self.gold_first_cluster[4], self.gold_first_cluster[0]): 1,
            (self.gold_first_cluster[4], self.gold_first_cluster[1]): 3,
            (self.gold_first_cluster[4], self.gold_first_cluster[2]): 1,
            (self.gold_first_cluster[4], self.gold_first_cluster[3]): 1,
            (self.gold_first_cluster[5], self.gold_first_cluster[0]): 1,
            (self.gold_first_cluster[5], self.gold_first_cluster[1]): 2,
            (self.gold_first_cluster[5], self.gold_first_cluster[2]): 3,
            (self.gold_first_cluster[5], self.gold_first_cluster[3]): 4,
            (self.gold_first_cluster[5], self.gold_first_cluster[4]): 5,
        })

        gold_graph = data_structures.EntityGraph.from_mentions(
            self.gold_first_cluster, "annotated_set_id", scores)[0]

        spanning_tree_edges = [
            (self.gold_first_cluster[1], self.gold_first_cluster[0]),
            (self.gold_first_cluster[2], self.gold_first_cluster[0]),
            (self.gold_first_cluster[3], self.gold_first_cluster[1]),
            (self.gold_first_cluster[4], self.gold_first_cluster[3]),
            (self.gold_first_cluster[5], self.gold_first_cluster[4]),
        ]

        self.assertEqual(
            spanning_tree_edges,
            spanning_tree_algorithms.recall_scores(
                gold_graph,
                gold_graph.partition(
                    data_structures.EntityGraph.from_mentions(
                        self.system1_mentions, "set_id", scores))))


if __name__ == '__main__':
    unittest.main()
