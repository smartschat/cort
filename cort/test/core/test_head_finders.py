import unittest

import nltk

from cort.core import head_finders
from cort.core import spans


__author__ = 'smartschat'


class TestHeadFinder(unittest.TestCase):
    def setUp(self):
        self.head_finder = head_finders.HeadFinder()

    def test_get_head_np(self):
        self.assertEqual(nltk.ParentedTree("NNS", ["police"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(NP (JJ Local) (NNS police))")))
        self.assertEqual(nltk.ParentedTree("NN", ["shop"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(NP (JJ Local) (NN shop))")))
        self.assertEqual(nltk.ParentedTree("NNP", ["NBC"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(NP (NNP NBC) (POS 's))")))
        self.assertEqual(nltk.ParentedTree("NN", ["wedding"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(NP (NP (NP (PRP$ his) (NN brother) (POS 's)) (NN wedding)) (PP (IN in) (NP (NNP Khan) (NNPS Younes))))")))
        self.assertEqual(nltk.ParentedTree("NNP", ["Taiwan"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(NP (NNP Taiwan) (POS 's))")))
        self.assertEqual(nltk.ParentedTree("NN", ["port"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(NP (NP (NP (NNP Yemen) (POS 's)) (NN port)) (PP (IN of) (NP (NNP Aden))))")))

    def test_get_head_vp(self):
        self.assertEqual(nltk.ParentedTree("VB", ["shoot"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(VP (VB shoot))")))

    def test_get_head_nml(self):
        self.assertEqual(nltk.ParentedTree("NN", ["curtain"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(NML (NN air) (NN curtain))")))

    def test_get_head_adjp(self):
        self.assertEqual(nltk.ParentedTree("JJ" ,["twelfth"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(ADJP (JJ twelfth) (CC and) (JJ thirteenth))")))

    def test_get_head_qp(self):
        self.assertEqual(nltk.ParentedTree("CD", ["forty"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(QP (CD forty) (HYPH -) (CD five))")))

    def test_get_head_whadvp(self):
        self.assertEqual(nltk.ParentedTree("WRB", ["how"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(WHADVP (WRB how))")))

    def test_get_head_s(self):
        parse = """(S
  (S
    (NP
      (NP
        (DT The)
        (ADJP (RBS most) (JJ important))
        (JJ Taiwanese)
        (JJ musical)
        (NN master))
      (PP (IN of) (NP (DT the) (JJ last) (JJ half) (NN century)))))
  (, ,)
  (NP (PRP he))
  (VP
    (VBD was)
    (NP
      (NP (DT a) (JJ beloved) (NN teacher))
      (PP (IN to) (NP (JJ many)))))
  (. .))"""

        self.assertEqual(nltk.ParentedTree("VBD", ["was"]), self.head_finder.get_head(nltk.ParentedTree.fromstring(parse)))

        parse_2 = "(S (`` `) (NP (NNP Bus) (NNP Stop) (POS ')))"

        self.assertEqual(nltk.ParentedTree("NNP", ["Stop"]),
                         self.head_finder.get_head(
                             nltk.ParentedTree.fromstring(parse_2)))

    def test_get_head_advp(self):
        self.assertEqual(nltk.ParentedTree("RB", ["here"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(ADVP (RB here))")))

    def test_get_head_whnp(self):
        self.assertEqual(nltk.ParentedTree("WP", ["who"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(WHNP (WP who))")))

    def test_get_head_sbar(self):
        parse = """(SBAR
  (WHNP (WP who))
  (S
    (VP
      (VBD had)
      (VP
        (VBN had)
        (NP (NP (JJ enough)) (PP (IN of) (NP (NN schooling))))))))"""

        self.assertEqual(nltk.ParentedTree("WP", ["who"]), self.head_finder.get_head(nltk.ParentedTree.fromstring(parse)))

    def test_get_head_pp(self):
        parse = """(PP
  (IN of)
  (NP
    (NP (NNS thousands))
    (PP (IN of) (NP (JJ non-profit) (NNS institutions)))))"""

        self.assertEqual(nltk.ParentedTree("IN", ["of"]), self.head_finder.get_head(nltk.ParentedTree.fromstring(parse)))

    def test_get_head_intj(self):
        self.assertEqual(nltk.ParentedTree("UH", ["oh"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(INTJ (UH oh) (PRP$ my) (NNP god))")))

    def test_get_head_sq(self):
        self.assertEqual(nltk.ParentedTree("VBP", ["are"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(SQ (VBP are) (NP (PRP they)) (NP (DT all) (NNS liars)))")))

    def test_get_head_ucp(self):
        self.assertEqual(nltk.ParentedTree("NN", ["trade"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(UCP (JJ economic) (CC and) (NN trade))")))

    def test_get_head_x(self):
        self.assertEqual(nltk.ParentedTree(":", ["--"]), self.head_finder.get_head(nltk.ParentedTree.fromstring("(X (NNS Men) (CC or) (: --))")))

    def test_get_head_sbarq(self):
        parse = """(SBARQ
  (WHADVP (WRB Where))
  (SQ (MD Should) (NP (NNP Chinese) (NNP Music)) (VP (VB Go)))
  (. ?))"""

        self.assertEqual(nltk.ParentedTree("MD", ["Should"]), self.head_finder.get_head(nltk.ParentedTree.fromstring(parse)))

    def test_get_head_frag(self):
        parse = """(FRAG
  (PP (IN On) (NP (DT the) (NN internet) (NN type)))
  (NP (NNP Iraq))
  (: :)
  (NP (NNP Beyond) (NNP Abu) (NNP Ghraib))
  (: :)
  (NP
    (NP (NN Detention) (CC and) (NN torture))
    (PP (IN in) (NP (NNP Iraq))))
  (. .))"""

        self.assertEqual(nltk.ParentedTree(".", ["."]), self.head_finder.get_head(
            nltk.ParentedTree.fromstring(
            parse)))

    def test_adjust_head_for_nam(self):
        self.assertEqual((spans.Span(0, 1), ["Khan", "Younes"]), head_finders.HeadFinder.adjust_head_for_nam(
            ["Khan", "Younes", "in", "the", "southern", "Ghaza", "Strip"],
            ["NNP", "NNS", "IN", "DT", "JJ", "NNP", "NNP"],
            "GPE"
        ))

        self.assertEqual((spans.Span(0, 1), ["Walter", "Sisulu"]), head_finders.HeadFinder.adjust_head_for_nam(
            ["Walter", "Sisulu"],
            ["NNP", "NNP"],
            "PERSON"
        ))

        self.assertEqual((spans.Span(1, 5), ['vice', 'president', 'Robert', 'W.', 'Reedy']), head_finders.HeadFinder.adjust_head_for_nam(
            ['former', 'vice', 'president', 'Robert', 'W.', 'Reedy'],
            ["JJ", "NN", "NN", "NNP", "NNP", "NNP"],
            "PERSON"
        ))

        self.assertEqual((spans.Span(0, 1), ['Michael', 'Wolf']), head_finders.HeadFinder.adjust_head_for_nam(
            ['Michael', 'Wolf', ',', 'a', 'contributing', 'editor'],
            ["NNP", "NNP", ",", "DT", "VBG", "NN"],
            "PERSON"
        ))

        self.assertEqual((spans.Span(0, 1), ['Mr.', 'Clinton']), head_finders.HeadFinder.adjust_head_for_nam(
            ['Mr.', 'Clinton'],
            ["NNP", "NNP"],
            "NONE"
        ))

        self.assertEqual((spans.Span(0, 0), ['Taiwan']), head_finders.HeadFinder.adjust_head_for_nam(
            ['Taiwan', "'s"],
            ["NNP", "POS"],
            "GPE"
        ))

        self.assertEqual((spans.Span(0, 2), ["Jim", "'s", "Tools"]), head_finders.HeadFinder.adjust_head_for_nam(
            ['Jim', "'s", "Tools"],
            ["NNP", "POS", "NNP"],
            "ORG"
        ))

        self.assertEqual((spans.Span(0, 3), ["Taiwan", "'s", "False", "Cypresses"]), head_finders.HeadFinder.adjust_head_for_nam(
            ["Taiwan", "'s", "False", "Cypresses"],
            ["NNP", "POS", "JJ", "NNP"],
            "NONE"
        ))

    def test_head_rule_cc(self):
        parse = """(NP
        (NP
            (NNS ruin))
        (CC and)
        (NP
            (NNS terror)))
        """

        self.assertEqual(nltk.ParentedTree("CC", ["and"]),
                         self.head_finder.get_head(nltk.ParentedTree.fromstring(
                parse)))

    def test_get_difficult_heads(self):
        parse = """(NP
  (S
    (VP
      (VP
        (VBG recalling)
        (NP (DT the) (JJ Korean) (NN delegation))
        (PP
          (IN to)
          (NP
            (DT the)
            (NNP Korean)
            (NML (NNP Military) (NNP Armistice))
            (NNP Commission))))
      (CC and)
      (VP
        (VBG setting)
        (PRT (RP up))
        (NP
          (NP
            (DT the)
            (NNP Panmunjom)
            (NNP Representative)
            (NNP Office))
          (PP
            (IN of)
            (NP
              (NP (DT the) (NNP Korean) (NNPS People) (POS 's))
              (NNP Army))))
        (PP (IN as) (NP (DT the) (JJ negotiatory) (NN organization))))))
  (, ,)
  (ADVP (FW etc)))"""

        parse2 = """(NP
  (QP (NNS Tens) (IN of) (NNS thousands))
  (PP (IN of) (NP (NNS people))))"""

        parse3 = """(NP
  (PRP he)
  (PRN
    (-LRB- -LRB-)
    (NP
      (NP (DT the) (NN one))
      (SBAR
        (WHNP (WP who))
        (S (VP (VBD tricked) (NP (DT these) (NNS people))))))
    (-RRB- -RRB-)))"""

        parse4 = """(UCP
  (NP (NN %um))
  (CC and)
  (S (NP (PRP you)) (VP (MD can) (VP (ADVP (RB also))))))"""

        self.assertEqual(nltk.ParentedTree("FW", ["etc"]), self.head_finder.get_head(nltk.ParentedTree.fromstring(parse)))
        self.assertEqual(nltk.ParentedTree("NNS", ["Tens"]), self.head_finder.get_head(nltk.ParentedTree.fromstring(parse2)))
        self.assertEqual(nltk.ParentedTree("-LRB-", ["-LRB-"]), self.head_finder.get_head(nltk.ParentedTree.fromstring(parse3)))
        self.assertEqual(nltk.ParentedTree("MD", ["can"]), self.head_finder.get_head(nltk.ParentedTree.fromstring(parse4)))

if __name__ == '__main__':
    unittest.main()
