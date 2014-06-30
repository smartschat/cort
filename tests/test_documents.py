from collections import defaultdict
from mentions import Mention
from spans import Span
from documents import CoNLLDocument
import unittest


__author__ = 'smartschat'


class TestDocuments(unittest.TestCase):
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
test2	0	3	the NN   *	-   -   -   -   -   (2|(0
test2	0	4	scorer  NN   *	-   -   -   -   -   2)|4)
test2	0	5	works   NN   *	-   -   -   -   -   0)
test2	0	6	.   NN   *)	-   -   -   -   -   -

#end	document"""

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

        self.yemen_example = """#begin document (bn/abc/00/abc_0030); part 000
bn/abc/00/abc_0030      0       0       Intelligence    NN      (TOP(S(NP*      -       -       -       -       *       (ARG0*  *       -
bn/abc/00/abc_0030      0       1       sources NNS     *)      source  -       3       -       *       *)      *       -
bn/abc/00/abc_0030      0       2       say     VBP     (VP*    say     01      1       -       *       (V*)    *       -
bn/abc/00/abc_0030      0       3       the     DT      (SBAR(S(NP*     -       -       -       -       *       (ARG1*  (ARG1*  -
bn/abc/00/abc_0030      0       4       target  NN      *)      target  -       2       -       *       *       *)      -
bn/abc/00/abc_0030      0       5       was     VBD     (VP*    be      -       1       -       *       *       *       -
bn/abc/00/abc_0030      0       6       to      TO      (S(VP*  -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       7       be      VB      (VP*    be      01      1       -       *       *       (V*)    -
bn/abc/00/abc_0030      0       8       a       DT      (NP(NP* -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       9       destroyer       NN      *)      -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       10      ,       ,       *       -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       11      the     DT      (NP*    -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       12      ``      ``      *       -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       13      USS     NNP     (NP*    -       -       -       -       (PRODUCT*       *       *       -
bn/abc/00/abc_0030      0       14      The     NNP     *       -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       15      Sullivans       NNP     *)      -       -       -       -       *)      *       *       -
bn/abc/00/abc_0030      0       16      ,       ,       *       -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       17      ''      ''      *       -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       18      which   WDT     (SBAR(WHNP*)    -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       19      refueled        VBD     (S(VP*  refuel  -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       20      in      IN      (PP*    -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       21      Yemen   NNP     (NP(NP(NP*      -       -       -       -       (GPE)   *       *       -
bn/abc/00/abc_0030      0       22      's      POS     *)      -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       23      port    NN      *)      port    -       1       -       (LOC*   *       *       -
bn/abc/00/abc_0030      0       24      of      IN      (PP*    -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       25      Aden    NNP     (NP*))))        -       -       -       -       *)      *       *       -
bn/abc/00/abc_0030      0       26      in      IN      (PP*    -       -       -       -       *       *       *       -
bn/abc/00/abc_0030      0       27      January NNP     (NP*))))))))))))))      -       -       -       -       (DATE)  *)      *       -
bn/abc/00/abc_0030      0       28      .       .       *))     -       -       -       -       *       *       *       -

#end document"""

        self.real_document = CoNLLDocument(self.real_example)
        self.complicated_mention_document = CoNLLDocument(self.complicated_mention_example)
        self.another_real_document = CoNLLDocument(self.another_real_example)
        self.yemen_document = CoNLLDocument(self.yemen_example)

    def test_get_id(self):
        self.assertEqual("voa_0220", self.real_document.id)

    def test_extract_id(self):
        self.assertEqual("voa_0220", CoNLLDocument.extract_id("#begin document (bn/voa/02/voa_0220); part 000"))

    def test_get_part(self):
        self.assertEqual(0, self.real_document.part)

    def test_extract_part(self):
        self.assertEqual(0, CoNLLDocument.extract_part("#begin document (bn/voa/02/voa_0220); part 000"))

    def test_extract_folder(self):
        self.assertEqual("bn/voa/02/", CoNLLDocument.extract_folder("#begin document (bn/voa/02/voa_0220); part 000"))

    def test_get_tokens(self):
        tokens = ["Unidentified","gunmen","in","north","western","Colombia","have","massacred","at","least","twelve",\
                  "peasants","in","the","second","such","incident","in","as","many","days",".","Local","police","say",\
                  "it","'s","not","clear","who","was","responsible","for","the","massacre","."]
        self.assertEqual(tokens, self.real_document.tokens)

    def test_get_ner(self):
        ner = ["NONE"] * 36
        ner[5:6] = ["GPE"]
        ner[8:11] = ["CARDINAL"]*3
        ner[14:15] = ["ORDINAL"]
        ner[18:21] = ["DATE"]*3

        self.assertEqual(ner, self.real_document.ner)

    def test_get_coref(self):
        simple = {
            Span(13, 20):  0,
            Span(33, 34): 0
        }

        complicated = {
            Span(0, 0): 0,
            Span(3, 3): 1,
            Span(3, 4): 0,
            Span(6, 6): 1,
            Span(6, 10): 4,
            Span(8, 8): 2,
            Span(9, 10): 2,
            Span(9, 11): 0
        }

        self.assertEqual(simple, self.real_document.coref)
        self.assertEqual(complicated, self.complicated_mention_document.coref)

    def test_extract_sentence_spans(self):
        sentence_spans_to_ids = {Span(0, 21): 0, Span(22, 35): 1}

        self.assertEqual(sentence_spans_to_ids, self.real_document.extract_sentence_spans())

    def test_get_embedding_sentence(self):
        expected = Span(22, 35)
        self.assertEqual(expected, self.real_document.get_embedding_sentence(Span(23, 24)))

    def test_get_parse(self):
        expected = "(TOP (S (NP (JJ Local) (NNS police)) (VP (VBP say) (SBAR (S (NP (PRP it)) (VP (VBZ 's) (RB not) \
(ADJP (JJ clear)) (SBAR (WHNP (WP who)) (S (VP (VBD was) (ADJP (JJ responsible) (PP (IN for) (NP \
(DT the) (NN massacre))))))))))) (. .)))"
        self.assertEqual(expected, self.real_document.get_parse(Span(22, 35)))

    def test_get_span_to_id(self):
        column = ["(1", "(1", "1)", "1)"]
        expected = defaultdict(list)
        expected[Span(0, 3)] = [1]
        expected[Span(1, 2)] = [1]

        self.assertEqual(expected, CoNLLDocument.get_span_to_id(column))

    def test_parse_span(self):
        column = ["(8", "-", "8)", "(9)", "(8)", "(5", "-", "-", "5)", "(9)|(10)", "(10)|(12)", "(12)", "(5", "-", "-", "5)"]
        self.assertEqual(None, CoNLLDocument.parse_span(Span(0, 1), column))
        self.assertEqual(5, CoNLLDocument.parse_span(Span(5, 8), column))
        self.assertEqual(12, CoNLLDocument.parse_span(Span(11, 11), column))

    def test_get_genre(self):
        self.assertEqual("bn", self.real_document.get_genre())
        self.assertEqual("unknown", self.complicated_mention_document.get_genre())

    def test_extract_system_mention_spans(self):
        expected_spans = sorted([
            Span(0, 1),
            Span(0, 5),
            Span(3, 5),
            Span(5, 5),
            Span(8, 10),
            Span(8, 11),
            Span(13, 16),
            Span(13, 20),
            Span(14, 14),
            Span(18, 20),
            Span(22, 23),
            Span(25, 25),
            Span(33, 34)
        ])

        self.assertEqual(expected_spans, self.real_document.extract_system_mention_spans())

        expected_spans = sorted([
            Span(2, 2),
            Span(4, 4),
            Span(6, 7),
            Span(6, 11),
            Span(9, 10),
            Span(9, 11)
        ])

        self.assertEqual(expected_spans, self.another_real_document.extract_system_mention_spans())

    def test_extract_system_mentions(self):
        expected_spans = sorted([
            Span(2, 2),
            Span(4, 4),
            Span(6, 11),
            Span(9, 10),
            Span(9, 11)
        ])

        self.another_real_document.extract_system_mentions()

        self.assertEqual(expected_spans, sorted([mention.span for mention in self.another_real_document.system_mentions]))

    def test_get_string_representation_of_mentions(self):
        expected = ["(0|(1)", "0)", "-", "-", "(1", "(2|1)", "2)", "(3)", "(3)", "-", "-", "-", "-"]
        mentions = [
            Mention(self.complicated_mention_document, Span(0, 0), {"annotated_set_id": 1}),
            Mention(self.complicated_mention_document, Span(0, 1), {"annotated_set_id": 0}),
            Mention(self.complicated_mention_document, Span(4, 5), {"annotated_set_id": 1}),
            Mention(self.complicated_mention_document, Span(5, 6), {"annotated_set_id": 2}),
            Mention(self.complicated_mention_document, Span(7, 7), {"annotated_set_id": 3}),
            Mention(self.complicated_mention_document, Span(8, 8), {"annotated_set_id": 3}),
        ]

        self.assertEqual(expected, self.complicated_mention_document.get_string_representation_of_mentions(12, mentions))

    def test_get_string_representation_of_mentions(self):
        expected = ["(0|(1)", "0)", "-", "-", "(1", "(2|1)", "2)", "(3)", "(3)", "-", "-", "-", "-"]
        mentions = [
            Mention(self.complicated_mention_document, Span(0, 0), {"set_id": 1}),
            Mention(self.complicated_mention_document, Span(0, 1), {"set_id": 0}),
            Mention(self.complicated_mention_document, Span(4, 5), {"set_id": 1}),
            Mention(self.complicated_mention_document, Span(5, 6), {"set_id": 2}),
            Mention(self.complicated_mention_document, Span(7, 7), {"set_id": 3}),
            Mention(self.complicated_mention_document, Span(8, 8), {"set_id": 3}),
        ]

        self.assertEqual(expected, self.complicated_mention_document.get_string_representation_of_mentions(12, mentions))

    def test_get_new_table_repr(self):
        expected = """test2	0	0	This	NN	(NP*	-	-	-	-	-	(0|(1)
test2	0	1	is	NN	*	-	-	-	-	-	0)
test2	0	2	just	NN	*	-	-	-	-	-	-
test2	0	3	a	NN	*	-	-	-	-	-	-
test2	0	4	test	NN	*	-	-	-	-	-	(1
test2	0	5	.	NN	*)	-	-	-	-	-	(2|1)

test2	0	0	It	NN	(NP*	-	-	-	-	-	2)
test2	0	1	shows	NN	*	-	-	-	-	-	(3)
test2	0	2	that	NN	*	-	-	-	-	-	(3)
test2	0	3	the	NN	*	-	-	-	-	-	-
test2	0	4	scorer	NN	*	-	-	-	-	-	-
test2	0	5	works	NN	*	-	-	-	-	-	-
test2	0	6	.	NN	*)	-	-	-	-	-	-"""

        mentions = [
            Mention(self.complicated_mention_document, Span(0, 0), {"set_id": 1}),
            Mention(self.complicated_mention_document, Span(0, 1), {"set_id": 0}),
            Mention(self.complicated_mention_document, Span(4, 5), {"set_id": 1}),
            Mention(self.complicated_mention_document, Span(5, 6), {"set_id": 2}),
            Mention(self.complicated_mention_document, Span(7, 7), {"set_id": 3}),
            Mention(self.complicated_mention_document, Span(8, 8), {"set_id": 3}),
        ]

        self.assertEqual(expected, self.complicated_mention_document.get_new_table_repr(mentions))

    def test_are_coreferent(self):
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
test2	0	3	the NN   *	-   -   -   -   -   (2|(0
test2	0	4	scorer  NN   *	-   -   -   -   -   2)|4)
test2	0	5	works   NN   *	-   -   -   -   -   0)
test2	0	6	.   NN   *)	-   -   -   -   -   -

#end	document"""

        self.assertEqual(True,
            self.complicated_mention_document.are_coreferent(
            Mention(self.complicated_mention_document, Span(0, 0), {"set_id": 1}),
            Mention(self.complicated_mention_document, Span(3, 4), {"set_id": 0}),
        ))

        self.assertEqual(False,
            self.complicated_mention_document.are_coreferent(
            Mention(self.complicated_mention_document, Span(0, 0), {"set_id": 1}),
            Mention(self.complicated_mention_document, Span(4, 4), {"set_id": 0}),
        ))

        self.assertEqual(False,
            self.complicated_mention_document.are_coreferent(
            Mention(self.complicated_mention_document, Span(3, 3), {"set_id": 1}),
            Mention(self.complicated_mention_document, Span(8, 8), {"set_id": 1}),
        ))


if __name__ == '__main__':
    unittest.main()
