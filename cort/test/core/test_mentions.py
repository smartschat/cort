import unittest

from cort.core.mentions import Mention
from cort.core.spans import Span
from cort.core.documents import CoNLLDocument


__author__ = 'smartschat'


class TestDiscourseUnits(unittest.TestCase):
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

        self.real_document = CoNLLDocument(self.real_example)

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

        self.for_head_example = """#begin document (wb/a2e/00/a2e_0000); part 000
wb/a2e/00/a2e_0000      0       0       Celebration     NN      (TOP(S(NP*      -       -       -       -       *       (ARG0*  -
wb/a2e/00/a2e_0000      0       1       Shooting        NN      *)      shoot   -       -       -       *       *)      -
wb/a2e/00/a2e_0000      0       2       Turns   VBZ     (VP*    turn    02      2       -       *       (V*)    -
wb/a2e/00/a2e_0000      0       3       Wedding NN      (NP*)   wed     -       -       -       *       (ARG1*) (3)
wb/a2e/00/a2e_0000      0       4       Into    IN      (PP*    -       -       -       -       *       (ARG2*  -
wb/a2e/00/a2e_0000      0       5       a       DT      (NP*    -       -       -       -       *       *       -
wb/a2e/00/a2e_0000      0       6       Funeral NN      *)      -       -       -       -       *       *       -
wb/a2e/00/a2e_0000      0       7       in      IN      (PP*    -       -       -       -       *       *       -
wb/a2e/00/a2e_0000      0       8       Southern        JJ      (NP*    -       -       -       -       *       *       (14
wb/a2e/00/a2e_0000      0       9       Gaza    NNP     *       -       -       -       -       (GPE*   *       -
wb/a2e/00/a2e_0000      0       10      Strip   NNP     *)))))) -       -       -       -       *)      *)      14)

#end document"""

        self.for_head_document = CoNLLDocument(self.for_head_example)

        self.date_mention_example = """#begin document (nw/wsj/24/wsj_2444); part 000
nw/wsj/24/wsj_2444   0   0    Employment    NN      (TOP(S(NP*)   employment  01   1   -           *    (V*)       (ARG1*)     -
nw/wsj/24/wsj_2444   0   1            is   VBZ            (VP*            be  01   1   -           *      *           (V*)     -
nw/wsj/24/wsj_2444   0   2           now    RB          (ADVP*)           -    -   -   -           *      *    (ARGM-TMP*)     -
nw/wsj/24/wsj_2444   0   3            4     CD  (ADJP(ADJP(QP*            -    -   -   -   (PERCENT*      *        (ARG2*      -
nw/wsj/24/wsj_2444   0   4             %    NN               *)           -    -   -   -           *)     *             *      -
nw/wsj/24/wsj_2444   0   5        higher   JJR               *)           -    -   -   -           *      *             *      -
nw/wsj/24/wsj_2444   0   6          than    IN            (PP*            -    -   -   -           *      *             *      -
nw/wsj/24/wsj_2444   0   7            in    IN            (PP*            -    -   -   -           *      *             *      -
nw/wsj/24/wsj_2444   0   8         1983     CD        (NP*)))))           -    -   -   -       (DATE)     *             *)   (16)
nw/wsj/24/wsj_2444   0   9             .     .              *))           -    -   -   -           *      *             *      -

bc/cctv/00/cctv_0000   0    0           For     IN  (TOP(S(PP*          -    -   -   Speaker#1        *      *    (ARGM-TMP*      -
bc/cctv/00/cctv_0000   0    1           two     CD        (NP*          -    -   -   Speaker#1   (DATE*      *             *      (1
bc/cctv/00/cctv_0000   0    2         years    NNS          *))         -    -   -   Speaker#1        *)     *             *)     1)
bc/cctv/00/cctv_0000   0    3             ,      ,           *          -    -   -   Speaker#1        *      *             *      -
bc/cctv/00/cctv_0000   0    4        Disney    NNP        (NP*)         -    -   -   Speaker#1     (ORG)     *        (ARG0*)   (12)
bc/cctv/00/cctv_0000   0    5           has    VBZ        (VP*        have  01   -   Speaker#1        *    (V*)            *      -
bc/cctv/00/cctv_0000   0    6    constantly     RB      (ADVP*)         -    -   -   Speaker#1        *      *    (ARGM-MNR*)     -
bc/cctv/00/cctv_0000   0    7    maintained    VBN        (VP*    maintain  01   1   Speaker#1        *      *           (V*)     -
bc/cctv/00/cctv_0000   0    8           its   PRP$        (NP*          -    -   -   Speaker#1        *      *        (ARG1*    (12)
bc/cctv/00/cctv_0000   0    9       mystery     NN         *)))         -    -   -   Speaker#1        *      *             *)     -
bc/cctv/00/cctv_0000   0   10             .      .          *))         -    -   -   Speaker#1        *      *             *      -

nw/wsj/24/wsj_2413   0    0    Government    NNP    (TOP(S(NP(NP*          -    -   -   -        *     (ARG0*        *       (ARG0*    (ARG0*         *    (16
nw/wsj/24/wsj_2413   0    1     officials    NNS                *)   official   -   1   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0    2          here     RB       (UCP(ADVP*)         -    -   -   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0    3           and     CC                *          -    -   -   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0    4            in     IN             (PP*          -    -   -   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0    5         other     JJ             (NP*          -    -   -   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0    6     countries    NNS             *))))    country   -   3   -        *          *)       *            *)        *)        *     16)
nw/wsj/24/wsj_2413   0    7          laid    VBD          (VP(VP*         lay  01   2   -        *        (V*)       *            *         *         *      -
nw/wsj/24/wsj_2413   0    8         plans    NNS          (NP(NP*)       plan   -   2   -        *     (ARG1*        *            *         *         *      -
nw/wsj/24/wsj_2413   0    9       through     IN             (PP*          -    -   -   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0   10           the     DT             (NP*          -    -   -   -   (DATE*          *        *            *         *         *     (6
nw/wsj/24/wsj_2413   0   11       weekend     NN               *))    weekend   -   -   -        *)         *        *            *         *         *      6)
nw/wsj/24/wsj_2413   0   12            to     TO           (S(VP*          -    -   -   -        *   (C-ARG1*        *            *         *         *      -
nw/wsj/24/wsj_2413   0   13          head     VB             (VP*        head  03   6   -        *          *      (V*)           *         *         *      -
nw/wsj/24/wsj_2413   0   14           off     RP            (PRT*)         -    -   -   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0   15             a     DT             (NP*          -    -   -   -        *          *   (ARG1*            *         *         *      -
nw/wsj/24/wsj_2413   0   16        Monday    NNP                *          -    -   -   -    (DATE)         *        *            *         *         *     (8)
nw/wsj/24/wsj_2413   0   17        market     NN                *      market   -   4   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0   18      meltdown     NN           *))))))         -    -   -   -        *         *))       *)           *         *         *      -
nw/wsj/24/wsj_2413   0   19           --       :                *          -    -   -   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0   20           but     CC                *          -    -   -   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0   21          went    VBD             (VP*          go  02   1   -        *          *        *          (V*)        *         *      -
nw/wsj/24/wsj_2413   0   22           out     IN             (PP*          -    -   -   -        *          *        *   (ARGM-DIR*         *         *      -
nw/wsj/24/wsj_2413   0   23            of     IN             (PP*          -    -   -   -        *          *        *            *         *         *      -
nw/wsj/24/wsj_2413   0   24         their   PRP$             (NP*          -    -   -   -        *          *        *            *         *         *    (16)
nw/wsj/24/wsj_2413   0   25           way     NN              *)))        way   -   5   -        *          *        *            *)        *         *      -
nw/wsj/24/wsj_2413   0   26            to     TO           (S(VP*          -    -   -   -        *          *        *   (ARGM-PNC*         *         *      -
nw/wsj/24/wsj_2413   0   27          keep     VB             (VP*        keep  02   1   -        *          *        *            *       (V*)        *      -
nw/wsj/24/wsj_2413   0   28         their   PRP$             (NP*          -    -   -   -        *          *        *            *         *    (ARG0*)   (16)
nw/wsj/24/wsj_2413   0   29         moves    NNS                *)       move  02   2   -        *          *        *            *         *       (V*)     -
nw/wsj/24/wsj_2413   0   30         quiet     JJ   (S(ADJP*)))))))         -    -   -   -        *          *        *            *)   (ARG1*)        *      -
nw/wsj/24/wsj_2413   0   31             .      .               *))         -    -   -   -        *          *        *            *         *         *      -

#end document"""

        self.date_mention_document = CoNLLDocument(self.date_mention_example)

    def test_mention_tokens(self):
        expected = ["the", "massacre"]
        self.assertEqual(
            expected,
            Mention.from_document(
                Span(33, 34),
                self.real_document).attributes["tokens"])

        expected = "the massacre"
        self.assertEqual(
            expected,
            Mention.from_document(
                Span(33, 34),
                self.real_document).attributes["tokens_as_lowercase_string"])

    def test_mention_type(self):
        self.assertEqual(
            "NAM",
            Mention.from_document(
                Span(37, 37),
                self.date_mention_document).attributes["type"])
        self.assertEqual(
            "NAM",
            Mention.from_document(
                Span(11, 12),
                self.date_mention_document).attributes["type"])
        self.assertEqual(
            "NAM",
            Mention.from_document(
                Span(31, 32),
                self.date_mention_document).attributes["type"])
        self.assertEqual(
            "NAM",
            Mention.from_document(
                Span(8, 8),
                self.date_mention_document).attributes["type"])
        self.assertEqual(
            "NOM",
            Mention.from_document(
                Span(33, 34),
                self.real_document).attributes["type"])

    def test_mention_set_id(self):
        self.assertEqual(
            0,
            Mention.from_document(
                Span(33, 34),
                self.real_document).attributes["annotated_set_id"])
        self.assertEqual(
            4,
            Mention.from_document(
                Span(6, 10),
                self.complicated_mention_document).attributes[
                    "annotated_set_id"])
        self.assertEqual(
            1,
            Mention.from_document(
                Span(3, 3),
                self.complicated_mention_document).attributes[
                    "annotated_set_id"])

    def test_mention_get_head(self):
        expected = ["massacre"]
        self.assertEqual(
            expected,
            Mention.from_document(
                Span(33, 34),
                self.real_document).attributes["head"])

        expected = ["Wedding"]
        self.assertEqual(
            expected,
            Mention.from_document(
                Span(3, 3),
                self.for_head_document).attributes["head"])

        expected = "wedding"
        self.assertEqual(
            expected,
            Mention.from_document(
                Span(3, 3),
                self.for_head_document).attributes["head_as_lowercase_string"])

    def test_mention_get_governor(self):
        expected = "massacred"
        self.assertEqual(
            expected,
            Mention.from_document(
                Span(0, 1),
                self.real_document).attributes["governor"])

    def test_mention_get_ancestry(self):
        expected = "-L-VBN-L-NONE"
        self.assertEqual(
            expected,
            Mention.from_document(
                Span(11, 11),
                self.real_document).attributes["ancestry"])

        expected = "-R-NNS-R-VBN"
        self.assertEqual(
            expected,
            Mention.from_document(
                Span(0, 0),
                self.real_document).attributes["ancestry"])

    def test_mention_get_head_span(self):
        self.assertEqual(
            Span(9, 10),
            Mention.from_document(
                Span(8, 10),
                self.for_head_document).attributes["head_span"])

    def test_mention_get_fine_type(self):
        self.assertEqual(
            "DEF",
            Mention.from_document(
                Span(33, 34),
                self.real_document).attributes["fine_type"])
        self.assertEqual(
            "DEF",
            Mention.from_document(
                Span(21, 27),
                self.date_mention_document).attributes["fine_type"])
        self.assertEqual(
            "INDEF",
            Mention.from_document(
                Span(22, 22),
                self.date_mention_document).attributes["fine_type"])
        self.assertEqual(
            "POSS_ADJ",
            Mention.from_document(
                Span(45, 45),
                self.date_mention_document).attributes["fine_type"])

    def test_mention_get_sentence_id(self):
        self.assertEqual(
            0,
            Mention.from_document(
                Span(13, 20),
                self.real_document).attributes["sentence_id"])
        self.assertEqual(
            1,
            Mention.from_document(
                Span(33, 34),
                self.real_document).attributes["sentence_id"])

    def test_mention_get_context(self):
        self.assertEqual(
            ["laid", "plans"],
            Mention.from_document(
                Span(21, 27),
                self.date_mention_document).get_context(2))
        self.assertEqual(
            None,
            Mention.from_document(
                Span(21, 27),
                self.date_mention_document).get_context(0))
        self.assertEqual(
            ["through"],
            Mention.from_document(
                Span(31, 32),
                self.date_mention_document).get_context(-1))
        self.assertEqual(
            None,
            Mention.from_document(
                Span(21, 27),
                self.date_mention_document).get_context(1000))

    def test_is_coreferent_with(self):

        self.assertEqual(True,
            Mention(
                None, Span(0, 0), {"annotated_set_id": 1}
            ).is_coreferent_with(
                Mention(None, Span(3, 4), {"annotated_set_id": 1})
            )
        )

        self.assertEqual(False,
            Mention(
                None, Span(0, 0), {"annotated_set_id": 1}
            ).is_coreferent_with(
                Mention(None, Span(3, 4), {"annotated_set_id": 0})
            )
        )

        self.assertEqual(False,
            Mention(
                None, Span(0, 0), {"annotated_set_id": None}
            ).is_coreferent_with(
                Mention(None, Span(3, 4), {"annotated_set_id": None})
            )
        )

        self.assertEqual(True,
            Mention(
                self.complicated_mention_document, Span(0, 0), {"annotated_set_id": 1}
            ).is_coreferent_with(
                Mention(self.complicated_mention_document, Span(3, 4), {"annotated_set_id": 1})
            )
        )

        self.assertEqual(False,
            Mention(
                self.complicated_mention_document, Span(0, 0),
                {"annotated_set_id": None}
            ).is_coreferent_with(
                Mention(self.complicated_mention_document, Span(3, 4),
                        {"annotated_set_id": None})
            )
        )

        self.assertEqual(False,
            Mention(
                self.complicated_mention_document, Span(0, 0), {"annotated_set_id": 1}
            ).is_coreferent_with(
                Mention(self.real_document, Span(13, 20),
                        {"annotated_set_id": 1})
            )
        )


if __name__ == '__main__':
    unittest.main()