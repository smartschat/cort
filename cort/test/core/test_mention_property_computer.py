import unittest


import nltk


from cort.core.spans import Span
from cort.core.documents import CoNLLDocument
from cort.core import mention_property_computer


__author__ = 'smartschat'


class TestMentionPropertyComputer(unittest.TestCase):
    def test_number(self):
        self.assertEqual(
            "SINGULAR",
            mention_property_computer.compute_number(
                {"tokens": ["him"], "pos": ["PRP"], "type": "PRO",
                 "citation_form": "he", "head_index": 0}))
        self.assertEqual(
            "SINGULAR",
            mention_property_computer.compute_number(
                {"tokens": ["the", "guy"], "pos": ["DT", "NN"], "type": "NOM",
                 "head_index": 1}))
        self.assertEqual(
            "PLURAL",
            mention_property_computer.compute_number(
                {"tokens": ["they"], "pos": ["PRP"],"type": "PRO",
                 "citation_form": "they", "head_index": 0}))
        self.assertEqual(
            "PLURAL",
            mention_property_computer.compute_number(
                {"tokens": ["these", "freaks"], "pos": ["DT", "NNS"],
                 "type": "NOM", "head_index": 1}))
        self.assertEqual(
            "PLURAL",
            mention_property_computer.compute_number(
                {"tokens": ["he", "and", "she"], "pos": ["PRP", "CC", "PRP"],
                 "type": "NOM", "head_index": 1}))

    def test_gender(self):
        self.assertEqual(
            "MALE",
            mention_property_computer.compute_gender(
                {"tokens": ["him"], "pos": ["PRP"], "type": "PRO",
                 "citation_form": "he", "head_index": 0}))
        self.assertEqual(
            "NEUTRAL",
            mention_property_computer.compute_gender(
                {"tokens": ["its"], "pos": ["PRP$"], "type": "PRO",
                 "citation_form": "it", "head_index": 0}))
        self.assertEqual(
            "FEMALE",
            mention_property_computer.compute_gender(
                {"tokens": ["Mrs.", "Robinson"], "pos": ["NNP", "NNP"],
                 "type": "NAM", "head_index": 1}))
        self.assertEqual(
            "MALE",
            mention_property_computer.compute_gender(
                {"tokens": ["Mr.", "FooBar"], "pos": ["NNP", "NNP"],
                 "type": "NAM", "head_index": 1}))
        self.assertEqual(
            "NEUTRAL",
            mention_property_computer.compute_gender(
                {"tokens": ["an", "arrow"], "head": ["arrow"],
                 "pos": ["DT", "NN"],"type": "NOM", "ner": ["-", "-"],
                 "head_index": 1}))
        self.assertEqual(
            "FEMALE",
            mention_property_computer.compute_gender(
                {"tokens": ["the", "girl"], "head": ["girl"],
                 "pos": ["DT", "NN"],"type": "NOM", "ner": ["-", "-"],
                 "head_index": 1}))
        self.assertEqual(
            "NEUTRAL",
            mention_property_computer.compute_gender(
                {"tokens": ["the", "shooting"], "head": ["shooting"],
                 "pos": ["DT", "NN"],"type": "NOM", "ner": ["-", "-"],
                 "head_index": 1}))
        self.assertEqual(
            "MALE",
            mention_property_computer.compute_gender(
                {"tokens": ["the", "groom"], "head": ["groom"],
                 "pos": ["DT", "NN"],"type": "NOM", "ner": ["-", "-"],
                 "head_index": 1}))
        self.assertEqual(
            "PLURAL",
            mention_property_computer.compute_gender(
                {"tokens": ["the", "guys"], "head": ["guys"],
                 "pos": ["DT", "NNS"],"type": "NOM", "ner": ["-", "-"],
                 "head_index": 1}))
        self.assertEqual(
            "NEUTRAL",
            mention_property_computer.compute_gender(
                {"tokens": ["the", "Mona", "Lisa"], "head": ["Mona", "Lisa"],
                 "pos": ["DT", "NNP", "NNP"],"type": "NAM",
                 "ner": ["-", "WORK_OF_ART", "WORK_OF_ART"], "head_index": 2}))

    def test_semantic_class(self):
        self.assertEqual(
            "PERSON",
            mention_property_computer.compute_semantic_class(
                {"tokens": ["him"], "pos": ["PRP"], "type": "PRO",
                 "citation_form": "he", "head_index": 0}))
        self.assertEqual(
            "OBJECT",
            mention_property_computer.compute_semantic_class(
                {"tokens": ["its"], "pos": ["PRP$"], "type": "PRO",
                 "citation_form": "it", "head_index": 0}))
        self.assertEqual(
            "PERSON",
            mention_property_computer.compute_semantic_class(
                {"tokens": ["Mrs.", "Robinson"], "ner": ["PERSON", "PERSON"],
                 "pos": ["NNP", "NNP"],"type": "NAM", "head_index": 1}))
        self.assertEqual(
            "OBJECT",
            mention_property_computer.compute_semantic_class(
                {"tokens": ["a", "house"], "head": ["house"],
                 "pos": ["DT", "NN"],"type": "NOM", "ner": ["NONE", "NONE"],
                 "head_index": 1}))
        self.assertEqual(
            "UNKNOWN",
            mention_property_computer.compute_semantic_class(
                {"tokens": ["adsfg"], "head": ["adsfg"],
                 "pos": ["NN"],"type": "NOM", "ner": ["NONE", "NONE"],
                 "head_index": 0}))

    def test_citation_form(self):
        self.assertEqual(
            "they",
            mention_property_computer.get_citation_form({"tokens": ["them"]}))
        self.assertEqual(
            "she",
            mention_property_computer.get_citation_form({"tokens": ["her"]}))
        self.assertEqual(
            None,
            mention_property_computer.get_citation_form({"tokens": ["why"]}))

    def test_get_head_index(self):
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

        real_document = CoNLLDocument(self.real_example)

        expected = 0
        head = nltk.ParentedTree.fromstring("(WHNP (WP who))")
        mention_subtree = mention_property_computer.get_relevant_subtree(
            Span(29, 34), real_document)
        self.assertEqual(expected, mention_property_computer.get_head_index(
            head, mention_subtree))

    def test_tree_is_apposition(self):
        self.assertEqual(
            True,
            mention_property_computer.is_apposition({
                "parse_tree": nltk.ParentedTree.fromstring(
                    "(NP (NP (NP (NNP Secretary)) (PP (IN of) (NP "
                    "(NNP State)))) (NP (NNP Madeleine) (NNP Albright)))")}))
        self.assertEqual(
            False,
            mention_property_computer.is_apposition({
                "parse_tree": nltk.ParentedTree.fromstring(
                    "(NP (NP (NNP Secretary)) (PP (IN of) "
                    "(NP (NNP State))))")}))
        self.assertEqual(
            False,
            mention_property_computer.is_apposition({
                "parse_tree": nltk.ParentedTree.fromstring(
                    "(NP (NP (NP (NNP Al) (NNP Gore) (POS 's)) (NN campaign) "
                    "(NN manager)) (, ,) (NP (NNP Bill) (NNP Daley)) (, ,))")}))
        self.assertEqual(
            False,
            mention_property_computer.is_apposition({
                "parse_tree": nltk.ParentedTree.fromstring(
                    "(NP (NP (NNS news)) (NP (CD today)))")}))
        self.assertEqual(
            False,
            mention_property_computer.is_apposition({
                "parse_tree": nltk.ParentedTree.fromstring(
                    "(NP (NP (PRP$ his) (NN brother)) (, ,) (NP (PRP$ his) "
                    "(NN sister)))")}))
        self.assertEqual(
            True,
            mention_property_computer.is_apposition({
                "parse_tree":  nltk.ParentedTree.fromstring(
                    "(NP (NP (NNP Barack) (NNP Obama)) (, ,) (NP (DT the) "
                    "(NN president)))")}))

if __name__ == '__main__':
    unittest.main()
