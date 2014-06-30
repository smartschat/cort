from nltk import ParentedTree
from mentions import Mention
from spans import Span
import mention_extractor
import unittest


__author__ = 'smartschat'


class TestMentionExtractor(unittest.TestCase):
    def setUp(self):
        self.tree = ParentedTree("(NP (NP (NP (PRP$ his) (NN brother) (POS 's)) (NN wedding)) (PP (IN in) (NP (NNP Khan) (NNPS Younes))))")

        self.proper_name_mention_tree = ParentedTree("(NP (NNP Taiwan) (POS 's))")
        self.proper_name_mention_ner = ["GPE", "NONE"]

        self.apposition_tree = ParentedTree("(NP (NP (NP (NNP Secretary)) (PP (IN of) (NP (NNP State)))) (NP (NNP Madeleine) (NNP Albright)))")
        self.apposition_ner = ["NONE", "NONE", "NONE", "PERSON", "PERSON"]

        self.more_proper_name_tree = ParentedTree("(NP (NP (DT the) (NNP General) (NNP Secretary)) (PP (IN of) (NP (DT the) (NNP CCP))))")
        self.more_proper_name_ner = ["NONE", "NONE", "NONE", "NONE", "NONE", "ORG"]

    def test_get_mentions_from_tree(self):
        expected_spans = sorted([
            Span(0, 6),
            Span(0, 3),
            Span(0, 2),
            Span(5, 6),
            Span(0, 0)]
        )

        self.assertEqual(expected_spans, mention_extractor.extract_mention_spans_from_tree(self.tree))

        expected_spans = sorted([
            Span(0, 0),
            Span(0, 2),
            Span(0, 4),
            Span(2, 2),
            Span(3, 4)]
        )

        self.assertEqual(expected_spans, mention_extractor.extract_mention_spans_from_tree(self.apposition_tree))

        expected_spans = sorted([
            Span(0, 0),
            Span(0, 2),
            Span(0, 4),
            Span(2, 2),
            Span(3, 4)]
        )

        self.assertEqual(expected_spans, mention_extractor.extract_mention_spans_from_tree(self.apposition_tree))

    def test_get_mention_spans_from_ner(self):
        self.assertEqual([Span(0, 1)], mention_extractor.get_span_from_ner([pos[1] for pos in self.proper_name_mention_tree.pos()], self.proper_name_mention_ner))
        self.assertEqual([Span(3, 4)], mention_extractor.get_span_from_ner([pos[1] for pos in self.apposition_tree.pos()], self.apposition_ner))

    def test_get_spans(self):
        subtree = self.tree[self.tree.treeposition_spanning_leaves(5, 7)]

        self.assertEqual(Span(5, 6), mention_extractor.get_in_tree_span(subtree))

    def test_post_process_same_head_largest_span(self):
        all_mentions = set(
            [Mention(None, Span(0, 3), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(3, 3)}),
             Mention(None, Span(0, 6), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(3, 3)}),
             Mention(None, Span(0, 2), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(1, 1)}),
             Mention(None, Span(5, 6), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(5, 6)}),
             Mention(None, Span(0, 0), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(0, 0)})])

        expected_mentions = sorted([
            Mention(None, Span(0, 6), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(3, 3)}),
            Mention(None, Span(0, 2), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(1, 1)}),
            Mention(None, Span(5, 6), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(5, 6)}),
            Mention(None, Span(0, 0), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(0, 0)})
        ])

        self.assertEqual(expected_mentions, mention_extractor.post_process_same_head_largest_span(all_mentions))

        all_mentions_2 = set(
            [Mention(None, Span(0, 1), {"tokens": ["Taiwan", "'s"], "type": "NAM", "head_index": 0, "head_span": Span(0, 0)}),
            Mention(None, Span(0, 0), {"tokens": ["Taiwan"], "type": "NAM", "head_index": 0, "head_span": Span(0, 0)}),
            Mention(None, Span(2, 3), {"tokens": ["the", "CCP"], "type": "NAM", "head_index": 1, "head_span": Span(3, 3)}),
            Mention(None, Span(3, 3), {"tokens": ["CCP"], "type": "NAM", "head_index": 0, "head_span": Span(3, 3)})])

        expected_mentions_2 = sorted([
            Mention(None, Span(0, 1), {"tokens": ["Taiwan", "'s"], "type": "NAM", "head_index": 0, "head_span": Span(0, 0)}),
            Mention(None, Span(2, 3), {"tokens": ["the", "CCP"], "type": "NAM", "head_index": 1, "head_span": Span(3, 3)}),
        ])

        self.assertEqual(expected_mentions_2, mention_extractor.post_process_same_head_largest_span(all_mentions_2))

    def test_post_process_embedded_head_largest_span(self):
        all_mentions_1 = set(
            [Mention(None, Span(0, 3), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(3, 3)}),
             Mention(None, Span(0, 6), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(2, 3)}),
             Mention(None, Span(0, 2), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(1, 1)}),
             Mention(None, Span(5, 6), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(5, 6)})])

        expected_mentions_1 = sorted([
            Mention(None, Span(0, 6), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(2, 3)}),
            Mention(None, Span(0, 2), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(1, 1)}),
            Mention(None, Span(5, 6), {"tokens": [], "type": "NOM", "head_index": 0, "head_span": Span(5, 6)})
        ])

        self.assertEqual(expected_mentions_1, mention_extractor.post_process_embedded_head_largest_span(all_mentions_1))

    def test_post_process_appositions(self):
        three_children_tree = ParentedTree("(NP (NP (NP (NP (DT The) (NNP ROC) (POS 's)) (NN ambassador)) (PP (IN to) (NP (NNP Nicaragua)))) (, ,) (NP (NNP Antonio) (NNP Tsai)) (, ,))")

        three_children_all_mentions = set(
            [Mention(None, Span(0, 6), {"tokens": ["The", "ROC", "'s", "ambassador", "to", "Nicaragua", ",", "Antonio", "Tsai"], "is_apposition": True, "type": "NAM", "parse_tree": three_children_tree}),
            Mention(None, Span(0, 4), {"tokens": ["The", "ROC", "'s", "ambassador", "to", "Nicaragua"], "is_apposition": False, "type": "NOM", "parse_tree": three_children_tree[0]}),
            Mention(None, Span(0, 3), {"tokens": ["The", "ROC", "'s", "ambassador"], "is_apposition": False, "type": "NOM", "parse_tree": three_children_tree[0][0]}),
            Mention(None, Span(0, 2), {"tokens": ["The", "ROC", "'s"], "is_apposition": False, "type": "NAM", "parse_tree": three_children_tree[0][0][0]}),
            Mention(None, Span(4, 4), {"tokens": ["Nicaragua"], "is_apposition": False, "type": "NAM", "parse_tree": three_children_tree[0][1][1]}),
            Mention(None, Span(5, 6), {"tokens": ["Antonio", "Tsai"], "is_apposition": False, "type": "NAM", "parse_tree": three_children_tree[2]})])

        three_children_expected = sorted([
            Mention(None, Span(0, 6), {"tokens": ["The", "ROC", "'s", "ambassador", "to", "Nicaragua", ",", "Antonio", "Tsai"], "is_apposition": True, "type": "NAM", "parse_tree": three_children_tree}),
            Mention(None, Span(0, 3), {"tokens": ["The", "ROC", "'s", "ambassador"], "is_apposition": False, "type": "NOM", "parse_tree": three_children_tree[0][0]}),
            Mention(None, Span(0, 2), {"tokens": ["The", "ROC", "'s"], "is_apposition": False, "type": "NAM", "parse_tree": three_children_tree[0][0][0]}),
            Mention(None, Span(4, 4), {"tokens": ["Nicaragua"], "is_apposition": False, "type": "NAM", "parse_tree": three_children_tree[0][1][1]}),
        ])

        self.assertEqual(three_children_expected, mention_extractor.post_process_appositions(three_children_all_mentions))

        two_children_tree = ParentedTree("(NP (NP (NP (NNP Secretary)) (PP (IN of) (NP (NNP State)))) (NP (NNP Madeleine) (NNP Albright)))")

        two_children_all_mentions = set(
            [Mention(None, Span(0, 4), {"tokens": ["Secretary", "of", "Sate", "Madeleine", "Albright"], "is_apposition": True, "type": "NAM", "parse_tree": two_children_tree}),
            Mention(None, Span(0, 0), {"tokens": ["Secretary"], "is_apposition": False, "type": "NAM", "parse_tree": two_children_tree[0][0]}),
            Mention(None, Span(0, 2), {"tokens": ["Secretary", "of", "State"], "is_apposition": False, "type": "NAM", "parse_tree": two_children_tree[0]}),
            Mention(None, Span(2, 2), {"tokens": ["State"], "is_apposition": False, "type": "NAM", "parse_tree": two_children_tree[0][1][1]}),
            Mention(None, Span(2, 2), {"tokens": ["Madeleine", "Albright"], "is_apposition": False, "type": "NAM", "parse_tree": two_children_tree[1]})])

        two_children_expected = sorted([
            Mention(None, Span(0, 4), {"tokens": ["Secretary", "of", "Sate", "Madeleine", "Albright"], "is_apposition": True, "type": "NAM", "parse_tree": two_children_tree})
        ])

        self.assertEqual(two_children_expected, mention_extractor.post_process_appositions(two_children_all_mentions))

if __name__ == '__main__':
    unittest.main()
