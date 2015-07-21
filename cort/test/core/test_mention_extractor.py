import unittest

import nltk

from cort.core import documents
from cort.core import mention_extractor
from cort.core import mentions
from cort.core import spans


__author__ = 'smartschat'


class TestMentionExtractor(unittest.TestCase):
    def setUp(self):
        self.real_example = """#begin document (bn/voa/02/voa_0220); part 000
bn/voa/02/voa_0220   0    0    Unidentified    JJ  (TOP(S(NP(NP*          -   -   -   -            *    -
bn/voa/02/voa_0220   0    1          gunmen   NNS              *)         -   -   -   -            *    -
bn/voa/02/voa_0220   0    2              in    IN           (PP*          -   -   -   -            *    -
bn/voa/02/voa_0220   0    3           north    JJ      (NP(ADJP*          -   -   -   -            *    -
bn/voa/02/voa_0220   0    4         western    JJ              *)         -   -   -   -            *    -
bn/voa/02/voa_0220   0    5        Colombia   NNP            *)))         -   -   -   -         (GPE)   -
bn/voa/02/voa_0220   0    6            have   VBP           (VP*        have  -   -   -            *    -
bn/voa/02/voa_0220   0    7       massacred   VBN           (VP*    massacre  -   -   -            *    -
bn/voa/02/voa_0220   0    8              at    IN   (NP(QP(ADVP*          -   -   -   -   (CARDINAL*    -
bn/voa/02/voa_0220   0    9           least   JJS              *)         -   -   -   -            *    -
bn/voa/02/voa_0220   0   10          twelve    CD              *)         -   -   -   -            *)   -
bn/voa/02/voa_0220   0   11        peasants   NNS              *)         -   -   -   -            *    -
bn/voa/02/voa_0220   0   12              in    IN           (PP*          -   -   -   -            *    -
bn/voa/02/voa_0220   0   13             the    DT        (NP(NP*          -   -   -   -            *   (0
bn/voa/02/voa_0220   0   14          second    JJ              *          -   -   -   -     (ORDINAL)   -
bn/voa/02/voa_0220   0   15            such    JJ              *          -   -   -   -            *    -
bn/voa/02/voa_0220   0   16        incident    NN              *)   incident  -   2   -            *    -
bn/voa/02/voa_0220   0   17              in    IN           (PP*          -   -   -   -            *    -
bn/voa/02/voa_0220   0   18              as    RB        (NP(QP*          -   -   -   -       (DATE*    -
bn/voa/02/voa_0220   0   19            many    JJ              *)         -   -   -   -            *    -
bn/voa/02/voa_0220   0   20            days   NNS         *))))))        day  -   4   -            *)   0)
bn/voa/02/voa_0220   0   21               .     .             *))         -   -   -   -            *    -

bn/voa/02/voa_0220   0    0          Local    JJ    (TOP(S(NP*          -    -   -   -   *   (ARG0*             *    -
bn/voa/02/voa_0220   0    1         police   NNS             *)     police   -   -   -   *        *)            *    -
bn/voa/02/voa_0220   0    2            say   VBP          (VP*         say  01   1   -   *      (V*)            *    -
bn/voa/02/voa_0220   0    3             it   PRP   (SBAR(S(NP*)         -    -   -   -   *   (ARG1*        (ARG1*)   -
bn/voa/02/voa_0220   0    4             's   VBZ          (VP*          be  01   1   -   *        *           (V*)   -
bn/voa/02/voa_0220   0    5            not    RB             *          -    -   -   -   *        *    (ARGM-NEG*)   -
bn/voa/02/voa_0220   0    6          clear    JJ        (ADJP*)         -    -   -   -   *        *        (ARG2*)   -
bn/voa/02/voa_0220   0    7            who    WP   (SBAR(WHNP*)         -    -   -   -   *        *             *    -
bn/voa/02/voa_0220   0    8            was   VBD        (S(VP*          be   -   1   -   *        *             *    -
bn/voa/02/voa_0220   0    9    responsible    JJ        (ADJP*          -    -   -   -   *        *             *    -
bn/voa/02/voa_0220   0   10            for    IN          (PP*          -    -   -   -   *        *             *    -
bn/voa/02/voa_0220   0   11            the    DT          (NP*          -    -   -   -   *        *             *   (0
bn/voa/02/voa_0220   0   12       massacre    NN    *))))))))))   massacre   -   -   -   *        *)            *    0)
bn/voa/02/voa_0220   0   13              .     .            *))         -    -   -   -   *        *             *    -

#end document
"""

        self.another_real_example = """#begin document (mz/sinorama/10/ectb_1050); part 006
mz/sinorama/10/ectb_1050        6       0       What    WP      (TOP(SBARQ(WHNP*)       -       -       -       -       *       (R-ARG1*)       -
mz/sinorama/10/ectb_1050        6       1       does    VBZ     (SQ*    do      -       7       -       *       *       -
mz/sinorama/10/ectb_1050        6       2       this    DT      (NP*)   -       -       -       -       *       (ARG0*) -
mz/sinorama/10/ectb_1050        6       3       tell    VB      (VP*    tell    01      1       -       *       (V*)    -
mz/sinorama/10/ectb_1050        6       4       us      PRP     (NP*)   -       -       -       -       *       (ARG2*) -
mz/sinorama/10/ectb_1050        6       5       about   IN      (PP*    -       -       -       -       *       (ARG1*  -
mz/sinorama/10/ectb_1050        6       6       the     DT      (NP(NP* -       -       -       -       *       *       -
mz/sinorama/10/ectb_1050        6       7       transformation  NN      *)      transformation  -       1       -       *       *       -
mz/sinorama/10/ectb_1050        6       8       of      IN      (PP*    -       -       -       -       *       *       -
mz/sinorama/10/ectb_1050        6       9       Taiwan  NNP     (NP(NP* -       -       -       -       (GPE)   *       -
mz/sinorama/10/ectb_1050        6       10      's      POS     *)      -       -       -       -       *       *       -
mz/sinorama/10/ectb_1050        6       11      townships       NNS     *)))))) township        -       1       -       *       *)      -
mz/sinorama/10/ectb_1050        6       12      ?       .       *))     -       -       -       -       *       *       -

#end	document"""

        self.real_document = documents.CoNLLDocument(self.real_example)
        self.another_real_document = documents.CoNLLDocument(
            self.another_real_example)

        self.tree = nltk.ParentedTree.fromstring(
            "(NP (NP (NP (PRP$ his) (NN brother) (POS 's)) (NN wedding)) "
            "(PP (IN in) (NP (NNP Khan) (NNPS Younes))))")

        self.proper_name_mention_tree = nltk.ParentedTree.fromstring(
            "(NP (NNP Taiwan) (POS 's))")
        self.proper_name_mention_ner = ["GPE", "NONE"]

        self.apposition_tree = nltk.ParentedTree.fromstring(
            "(NP (NP (NP (NNP Secretary)) (PP (IN of) (NP (NNP State)))) "
            "(NP (NNP Madeleine) (NNP Albright)))")

        self.apposition_ner = ["NONE", "NONE", "NONE", "PERSON", "PERSON"]

        self.more_proper_name_tree = nltk.ParentedTree.fromstring(
            "(NP (NP (DT the) (NNP General) (NNP Secretary)) (PP (IN of) "
            "(NP (DT the) (NNP CCP))))")

        self.more_proper_name_ner = ["NONE", "NONE", "NONE", "NONE", "NONE",
                                     "ORG"]

    def test_extract_system_mentions(self):
        expected_spans = sorted([
            spans.Span(0, 1),
            spans.Span(0, 5),
            spans.Span(3, 5),
            spans.Span(5, 5),
            spans.Span(8, 10),
            spans.Span(8, 11),
            spans.Span(13, 16),
            spans.Span(13, 20),
            spans.Span(14, 14),
            spans.Span(18, 20),
            spans.Span(22, 23),
            spans.Span(25, 25),
            spans.Span(33, 34)
        ])

        self.assertEqual(expected_spans,
                         [mention.span for
                          mention in mention_extractor.extract_system_mentions(
                             self.real_document, filter_mentions=False)[1:]])

        expected_spans = sorted([
            spans.Span(2, 2),
            spans.Span(4, 4),
            spans.Span(6, 7),
            spans.Span(6, 11),
            spans.Span(9, 10),
            spans.Span(9, 11)
        ])

        self.assertEqual(expected_spans,
                         [mention.span for
                          mention in mention_extractor.extract_system_mentions(
                             self.another_real_document,
                             filter_mentions=False)[1:]])

        expected_spans = sorted([
            spans.Span(2, 2),
            spans.Span(4, 4),
            spans.Span(6, 11),
            spans.Span(9, 10),
            spans.Span(9, 11)
        ])

        self.assertEqual(expected_spans,
                         [mention.span for
                          mention in mention_extractor.extract_system_mentions(
                             self.another_real_document,
                             filter_mentions=True)[1:]])

    def test_post_process_same_head_largest_span(self):
        all_mentions = {
            mentions.Mention(
                None,
                spans.Span(0, 3),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(3, 3)}),
            mentions.Mention(
                None,
                spans.Span(0, 6),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(3, 3)}),
            mentions.Mention(
                None,
                spans.Span(0, 2),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(1, 1)}),
            mentions.Mention(
                None,
                spans.Span(5, 6),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(5, 6)}),
            mentions.Mention(
                None,
                spans.Span(0, 0),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(0, 0)})}

        expected_mentions = sorted([
            mentions.Mention(
                None,
                spans.Span(0, 6),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(3, 3)}),
            mentions.Mention(
                None,
                spans.Span(0, 2),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(1, 1)}),
            mentions.Mention(
                None,
                spans.Span(5, 6),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(5, 6)}),
            mentions.Mention(
                None,
                spans.Span(0, 0),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(0, 0)})
        ])

        self.assertEqual(
            expected_mentions,
            mention_extractor.post_process_same_head_largest_span(all_mentions))

        all_mentions_2 = {
            mentions.Mention(
                None,
                spans.Span(0, 1),
                {"tokens": ["Taiwan", "'s"], "type": "NAM", "head_index": 0,
                 "head_span": spans.Span(0, 0)}),
            mentions.Mention(
                None,
                spans.Span(0, 0),
                {"tokens": ["Taiwan"], "type": "NAM", "head_index": 0,
                 "head_span": spans.Span(0, 0)}),
            mentions.Mention(
                None,
                spans.Span(2, 3),
                {"tokens": ["the", "CCP"], "type": "NAM", "head_index": 1,
                 "head_span": spans.Span(3, 3)}),
            mentions.Mention(
                None,
                spans.Span(3, 3),
                {"tokens": ["CCP"], "type": "NAM", "head_index": 0,
                 "head_span": spans.Span(3, 3)})}

        expected_mentions_2 = sorted([
            mentions.Mention(
                None,
                spans.Span(0, 1),
                {"tokens": ["Taiwan", "'s"], "type": "NAM", "head_index": 0,
                 "head_span": spans.Span(0, 0)}),
            mentions.Mention(
                None,
                spans.Span(2, 3),
                {"tokens": ["the", "CCP"], "type": "NAM", "head_index": 1,
                 "head_span": spans.Span(3, 3)}),
        ])

        self.assertEqual(
            expected_mentions_2,
            mention_extractor.post_process_same_head_largest_span(
                all_mentions_2))

    def test_post_process_embedded_head_largest_span(self):
        all_mentions_1 = {
            mentions.Mention(
                None,
                spans.Span(0, 3),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(3, 3)}),
            mentions.Mention(
                None,
                spans.Span(0, 6),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(2, 3)}),
            mentions.Mention(
                None,
                spans.Span(0, 2),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(1, 1)}),
            mentions.Mention(
                None,
                spans.Span(5, 6),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(5, 6)})
        }

        expected_mentions_1 = sorted([
            mentions.Mention(
                None,
                spans.Span(0, 6),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(2, 3)}),
            mentions.Mention(
                None,
                spans.Span(0, 2),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(1, 1)}),
            mentions.Mention(
                None,
                spans.Span(5, 6),
                {"tokens": [], "type": "NOM", "head_index": 0,
                 "head_span": spans.Span(5, 6)})
        ])

        self.assertEqual(
            expected_mentions_1,
            mention_extractor.post_process_embedded_head_largest_span(
                all_mentions_1))

    def test_post_process_appositions(self):
        three_children_tree = nltk.ParentedTree.fromstring(
            "(NP (NP (NP (NP (DT The) (NNP ROC) (POS 's)) (NN ambassador)) "
            "(PP (IN to) (NP (NNP Nicaragua)))) (, ,) (NP (NNP Antonio) "
            "(NNP Tsai)) (, ,))")

        three_children_all_mentions = {
            mentions.Mention(
                None,
                spans.Span(0, 6),
                {"tokens": ["The", "ROC", "'s", "ambassador", "to",
                            "Nicaragua", ",", "Antonio", "Tsai"],
                 "is_apposition": True, "type": "NAM",
                 "parse_tree": three_children_tree}),
            mentions.Mention(
                None,
                spans.Span(0, 4),
                {"tokens": ["The", "ROC", "'s", "ambassador", "to",
                            "Nicaragua"],
                 "is_apposition": False, "type": "NOM",
                 "parse_tree": three_children_tree[0]}),
            mentions.Mention(
                None,
                spans.Span(0, 3),
                {"tokens": ["The", "ROC", "'s", "ambassador"],
                 "is_apposition": False, "type": "NOM",
                 "parse_tree": three_children_tree[0][0]}),
            mentions.Mention(
                None,
                spans.Span(0, 2),
                {"tokens": ["The", "ROC", "'s"], "is_apposition": False,
                 "type": "NAM", "parse_tree": three_children_tree[0][0][0]}),
            mentions.Mention(
                None,
                spans.Span(4, 4),
                {"tokens": ["Nicaragua"], "is_apposition": False,
                 "type": "NAM", "parse_tree": three_children_tree[0][1][1]}),
            mentions.Mention(
                None,
                spans.Span(5, 6),
                {"tokens": ["Antonio", "Tsai"], "is_apposition": False,
                 "type": "NAM", "parse_tree": three_children_tree[2]})}

        three_children_expected = sorted([
            mentions.Mention(
                None,
                spans.Span(0, 6),
                {"tokens": ["The", "ROC", "'s", "ambassador", "to",
                            "Nicaragua", ",", "Antonio", "Tsai"],
                 "is_apposition": True, "type": "NAM",
                 "parse_tree": three_children_tree}),
            mentions.Mention(
                None,
                spans.Span(0, 3),
                {"tokens": ["The", "ROC", "'s", "ambassador"],
                 "is_apposition": False, "type": "NOM",
                 "parse_tree": three_children_tree[0][0]}),
            mentions.Mention(
                None,
                spans.Span(0, 2),
                {"tokens": ["The", "ROC", "'s"], "is_apposition": False,
                 "type": "NAM", "parse_tree": three_children_tree[0][0][0]}),
            mentions.Mention(
                None,
                spans.Span(4, 4),
                {"tokens": ["Nicaragua"], "is_apposition": False,
                 "type": "NAM", "parse_tree": three_children_tree[0][1][1]}),
        ])

        self.assertEqual(
            three_children_expected,
            mention_extractor.post_process_appositions(
                three_children_all_mentions))

        two_children_tree = nltk.ParentedTree.fromstring(
            "(NP (NP (NP (NNP Secretary)) (PP (IN of) (NP (NNP State)))) "
            "(NP (NNP Madeleine) (NNP Albright)))")

        two_children_all_mentions = {
            mentions.Mention(
                None,
                spans.Span(0, 4),
                {"tokens": ["Secretary", "of", "Sate", "Madeleine",
                            "Albright"],
                 "is_apposition": True, "type": "NAM",
                 "parse_tree": two_children_tree}),
            mentions.Mention(
                None,
                spans.Span(0, 0),
                {"tokens": ["Secretary"], "is_apposition": False,
                 "type": "NAM", "parse_tree": two_children_tree[0][0]}),
            mentions.Mention(
                None,
                spans.Span(0, 2),
                {"tokens": ["Secretary", "of", "State"],
                 "is_apposition": False, "type": "NAM",
                 "parse_tree": two_children_tree[0]}),
            mentions.Mention(
                None,
                spans.Span(2, 2),
                {"tokens": ["State"], "is_apposition": False,
                 "type": "NAM", "parse_tree": two_children_tree[0][1][1]}),
            mentions.Mention(
                None,
                spans.Span(2, 2),
                {"tokens": ["Madeleine", "Albright"], "is_apposition": False,
                 "type": "NAM", "parse_tree": two_children_tree[1]})}

        two_children_expected = sorted([
            mentions.Mention(
                None,
                spans.Span(0, 4),
                {"tokens": ["Secretary", "of", "Sate", "Madeleine",
                            "Albright"],
                 "is_apposition": True, "type": "NAM",
                 "parse_tree": two_children_tree})
        ])

        self.assertEqual(
            two_children_expected,
            mention_extractor.post_process_appositions(
                two_children_all_mentions))

if __name__ == '__main__':
    unittest.main()
