from collections import namedtuple
import unittest

from crtk.analysis import error_representations
from crtk.analysis import spanning_tree_algorithms
from crtk.core import corpora
from crtk.core import mentions
from crtk.core import spans

__author__ = 'smartschat'


class TestErrorAnalysis(unittest.TestCase):
    def setUp(self):
        self.first_cluster = [
            mentions.Mention(
                None,
                spans.Span(0, 0),
                {"tokens": ["a"], "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(1, 1),
                {"tokens": ["b"], "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(2, 3),
                {"tokens": ["c", "d"], "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(4, 5),
                {"tokens": ["e", "f"], "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(5, 6),
                {"tokens": ["f", "g"], "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(7, 7),
                {"tokens": ["h"], "annotated_set_id": 0}),
        ]

        self.second_cluster = [
            mentions.Mention(
                None,
                spans.Span(3, 4),
                {"tokens": ["d", "e"], "annotated_set_id": 1}),

            mentions.Mention(
                None,
                spans.Span(7, 8),
                {"tokens": ["h", "i"], "annotated_set_id": 1}),

            mentions.Mention(
                None,
                spans.Span(10, 10),
                {"tokens": ["k"], "annotated_set_id": 1})
        ]

        self.system_cluster = [
            mentions.Mention(
                None,
                spans.Span(0, 0),
                {"tokens": ["a"], "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(2, 3),
                {"tokens": ["c", "d"], "annotated_set_id": 0}),

            mentions.Mention(
                None,
                spans.Span(4, 5),
                {"tokens": ["e", "f"], "annotated_set_id": 2}),

            mentions.Mention(
                None,
                spans.Span(5, 6),
                {"tokens": ["f", "g"], "annotated_set_id": 2}),

            mentions.Mention(
                None,
                spans.Span(7, 7),
                {"tokens": ["h"], "annotated_set_id": 1}),

            mentions.Mention(
                None,
                spans.Span(10, 10),
                {"tokens": ["k"], "annotated_set_id": 1})
        ]

        self.maxDiff = None

    def test_compute_errors(self):
        # fake document using a named tuple
        document = namedtuple("Document", "annotated_mentions")
        doc_gold = document(self.first_cluster + self.second_cluster)
        doc_system = document(self.system_cluster)
        corpus_gold = corpora.Corpus("fake gold", [doc_gold])
        corpus_system = corpora.Corpus("fake system", [doc_system])

        ea = error_representations.ErrorAnalysis(
            corpus_gold,
            corpus_system,
            spanning_tree_algorithms.recall_closest,
            spanning_tree_algorithms.precision_system_output
        )

        self.assertEqual(
            error_representations.ErrorSet([
                (self.first_cluster[1], self.first_cluster[0]),
                (self.first_cluster[3], self.first_cluster[2]),
                (self.first_cluster[5], self.first_cluster[4]),
                (self.second_cluster[1], self.second_cluster[0]),
                (self.second_cluster[2], self.second_cluster[1]),
            ]),
            ea.recall_errors
        )

    if __name__ == '__main__':
        unittest.main()
