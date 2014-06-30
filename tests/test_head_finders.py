from nltk import Tree
from head_finders import HeadFinder
from spans import Span
import unittest


__author__ = 'smartschat'


class TestHeadFinder(unittest.TestCase):
    def test_get_head_np(self):
        self.assertEqual(Tree("NNS", ["police"]), HeadFinder.get_head(Tree("(NP (JJ Local) (NNS police))")))
        self.assertEqual(Tree("NN", ["shop"]), HeadFinder.get_head(Tree("(NP (JJ Local) (NN shop))")))
        self.assertEqual(Tree("NNP", ["NBC"]), HeadFinder.get_head(Tree("(NP (NNP NBC) (POS 's))")))
        self.assertEqual(Tree("NN", ["wedding"]), HeadFinder.get_head(Tree("(NP (NP (NP (PRP$ his) (NN brother) (POS 's)) (NN wedding)) (PP (IN in) (NP (NNP Khan) (NNPS Younes))))")))
        self.assertEqual(Tree("NNP", ["Taiwan"]), HeadFinder.get_head(Tree("(NP (NNP Taiwan) (POS 's))")))
        self.assertEqual(Tree("NN", ["port"]), HeadFinder.get_head(Tree("(NP (NP (NP (NNP Yemen) (POS 's)) (NN port)) (PP (IN of) (NP (NNP Aden))))")))

    def test_get_head_vp(self):
        self.assertEqual(Tree("VB", ["shoot"]), HeadFinder.get_head(Tree("(VP (VB shoot))")))

    def test_get_head_nml(self):
        self.assertEqual(Tree("NN", ["curtain"]), HeadFinder.get_head(Tree("(NML (NN air) (NN curtain))")))

    def test_get_head_adjp(self):
        self.assertEqual(Tree("JJ" ,["twelfth"]), HeadFinder.get_head(Tree("(ADJP (JJ twelfth) (CC and) (JJ thirteenth))")))

    def test_get_head_qp(self):
        self.assertEqual(Tree("CD", ["forty"]), HeadFinder.get_head(Tree("(QP (CD forty) (HYPH -) (CD five))")))

    def test_get_head_whadvp(self):
        self.assertEqual(Tree("WRB", ["how"]), HeadFinder.get_head(Tree("(WHADVP (WRB how))")))

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

        self.assertEqual(Tree("VBD", ["was"]), HeadFinder.get_head(Tree(parse)))

    def test_get_head_advp(self):
        self.assertEqual(Tree("RB", ["here"]), HeadFinder.get_head(Tree("(ADVP (RB here))")))

    def test_get_head_whnp(self):
        self.assertEqual(Tree("WP", ["who"]), HeadFinder.get_head(Tree("(WHNP (WP who))")))

    def test_get_head_sbar(self):
        parse = """(SBAR
  (WHNP (WP who))
  (S
    (VP
      (VBD had)
      (VP
        (VBN had)
        (NP (NP (JJ enough)) (PP (IN of) (NP (NN schooling))))))))"""

        self.assertEqual(Tree("WP", ["who"]), HeadFinder.get_head(Tree(parse)))

    def test_get_head_pp(self):
        parse = """(PP
  (IN of)
  (NP
    (NP (NNS thousands))
    (PP (IN of) (NP (JJ non-profit) (NNS institutions)))))"""

        self.assertEqual(Tree("IN", ["of"]), HeadFinder.get_head(Tree(parse)))

    def test_get_head_intj(self):
        self.assertEqual(Tree("UH", ["oh"]), HeadFinder.get_head(Tree("(INTJ (UH oh) (PRP$ my) (NNP god))")))

    def test_get_head_sq(self):
        self.assertEqual(Tree("VBP", ["are"]), HeadFinder.get_head(Tree("(SQ (VBP are) (NP (PRP they)) (NP (DT all) (NNS liars)))")))

    def test_get_head_ucp(self):
        self.assertEqual(Tree("NN", ["trade"]), HeadFinder.get_head(Tree("(UCP (JJ economic) (CC and) (NN trade))")))

    def test_get_head_x(self):
        self.assertEqual(Tree(":", ["--"]), HeadFinder.get_head(Tree("(X (NNS Men) (CC or) (: --))")))

    def test_get_head_sbarq(self):
        parse = """(SBARQ
  (WHADVP (WRB Where))
  (SQ (MD Should) (NP (NNP Chinese) (NNP Music)) (VP (VB Go)))
  (. ?))"""

        self.assertEqual(Tree("MD", ["Should"]), HeadFinder.get_head(Tree(parse)))

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

        self.assertEqual(Tree(".", ["."]), HeadFinder.get_head(Tree(parse)))

    def test_adjust_head_for_nam(self):
        self.assertEqual((Span(0, 1), ["Khan", "Younes"]), HeadFinder.adjust_head_for_nam(
            ["Khan", "Younes", "in", "the", "southern", "Ghaza", "Strip"],
            ["NNP", "NNS", "IN", "DT", "JJ", "NNP", "NNP"],
            "GPE"
        ))

        self.assertEqual((Span(0, 1), ["Walter", "Sisulu"]), HeadFinder.adjust_head_for_nam(
            ["Walter", "Sisulu"],
            ["NNP", "NNP"],
            "PERSON"
        ))

        self.assertEqual((Span(1, 5), ['vice', 'president', 'Robert', 'W.', 'Reedy']), HeadFinder.adjust_head_for_nam(
            ['former', 'vice', 'president', 'Robert', 'W.', 'Reedy'],
            ["JJ", "NN", "NN", "NNP", "NNP", "NNP"],
            "PERSON"
        ))

        self.assertEqual((Span(0, 1), ['Michael', 'Wolf']), HeadFinder.adjust_head_for_nam(
            ['Michael', 'Wolf', ',', 'a', 'contributing', 'editor'],
            ["NNP", "NNP", ",", "DT", "VBG", "NN"],
            "PERSON"
        ))

        self.assertEqual((Span(0, 1), ['Mr.', 'Clinton']), HeadFinder.adjust_head_for_nam(
            ['Mr.', 'Clinton'],
            ["NNP", "NNP"],
            "NONE"
        ))

        self.assertEqual((Span(0, 0), ['Taiwan']), HeadFinder.adjust_head_for_nam(
            ['Taiwan', "'s"],
            ["NNP", "POS"],
            "GPE"
        ))

        self.assertEqual((Span(0, 2), ["Jim", "'s", "Tools"]), HeadFinder.adjust_head_for_nam(
            ['Jim', "'s", "Tools"],
            ["NNP", "POS", "NNP"],
            "ORG"
        ))

        self.assertEqual((Span(0, 3), ["Taiwan", "'s", "False", "Cypresses"]), HeadFinder.adjust_head_for_nam(
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

        self.assertEqual(Tree("CC", ["and"]), HeadFinder.rule_CC(Tree(parse)))

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

        self.assertEqual(Tree("FW", ["etc"]), HeadFinder.get_head(Tree(parse)))
        self.assertEqual(Tree("NNS", ["Tens"]), HeadFinder.get_head(Tree(parse2)))
        self.assertEqual(Tree("-LRB-", ["-LRB-"]), HeadFinder.get_head(Tree(parse3)))
        self.assertEqual(Tree("MD", ["can"]), HeadFinder.get_head(Tree(parse4)))

if __name__ == '__main__':
    unittest.main()
