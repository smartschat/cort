import unittest


import nltk

from cort.core.mentions import Mention
from cort.core.spans import Span
from cort.core.documents import CoNLLDocument
from cort.core.mention_property_computer import EnglishMentionPropertyComputer
from cort.core.head_finders import EnglishHeadFinder


__author__ = 'smartschat'


class TestEnglishMentionPropertyComputer(unittest.TestCase):
    def setUp(self):
        self.property_computer = EnglishMentionPropertyComputer(
            EnglishHeadFinder()
        )

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

        self.real_document.enrich_with_parse_trees()
        self.real_document.enrich_with_dependency_trees()

        self.date_mention_document.enrich_with_parse_trees()
        self.date_mention_document.enrich_with_dependency_trees()

        self.for_head_document.enrich_with_parse_trees()
        self.for_head_document.enrich_with_dependency_trees()

    def test_number(self):
        self.assertEqual(
            "SINGULAR",
            self.property_computer._compute_number(
                {"tokens": ["him"], "pos": ["PRP"], "type": "PRO",
                 "citation_form": "he", "head_index": 0}))
        self.assertEqual(
            "SINGULAR",
            self.property_computer._compute_number(
                {"tokens": ["the", "guy"], "pos": ["DT", "NN"], "type": "NOM",
                 "head_index": 1}))
        self.assertEqual(
            "PLURAL",
            self.property_computer._compute_number(
                {"tokens": ["they"], "pos": ["PRP"],"type": "PRO",
                 "citation_form": "they", "head_index": 0}))
        self.assertEqual(
            "PLURAL",
            self.property_computer._compute_number(
                {"tokens": ["these", "freaks"], "pos": ["DT", "NNS"],
                 "type": "NOM", "head_index": 1}))
        self.assertEqual(
            "PLURAL",
            self.property_computer._compute_number(
                {"tokens": ["he", "and", "she"], "pos": ["PRP", "CC", "PRP"],
                 "type": "NOM", "head_index": 1}))

    def test_gender(self):
        self.assertEqual(
            "MALE",
            self.property_computer._compute_gender(
                {"tokens": ["him"], "pos": ["PRP"], "type": "PRO",
                 "citation_form": "he", "head_index": 0}))
        self.assertEqual(
            "NEUTRAL",
            self.property_computer._compute_gender(
                {"tokens": ["its"], "pos": ["PRP$"], "type": "PRO",
                 "citation_form": "it", "head_index": 0}))
        self.assertEqual(
            "FEMALE",
            self.property_computer._compute_gender(
                {"tokens": ["Mrs.", "Robinson"], "pos": ["NNP", "NNP"],
                 "type": "NAM", "head_index": 1}))
        self.assertEqual(
            "MALE",
            self.property_computer._compute_gender(
                {"tokens": ["Mr.", "FooBar"], "pos": ["NNP", "NNP"],
                 "type": "NAM", "head_index": 1}))
        self.assertEqual(
            "NEUTRAL",
            self.property_computer._compute_gender(
                {"tokens": ["an", "arrow"], "head": ["arrow"],
                 "pos": ["DT", "NN"],"type": "NOM", "ner": ["-", "-"],
                 "head_index": 1}))
        self.assertEqual(
            "FEMALE",
            self.property_computer._compute_gender(
                {"tokens": ["the", "girl"], "head": ["girl"],
                 "pos": ["DT", "NN"],"type": "NOM", "ner": ["-", "-"],
                 "head_index": 1}))
        self.assertEqual(
            "NEUTRAL",
            self.property_computer._compute_gender(
                {"tokens": ["the", "shooting"], "head": ["shooting"],
                 "pos": ["DT", "NN"],"type": "NOM", "ner": ["-", "-"],
                 "head_index": 1}))
        self.assertEqual(
            "MALE",
            self.property_computer._compute_gender(
                {"tokens": ["the", "groom"], "head": ["groom"],
                 "pos": ["DT", "NN"],"type": "NOM", "ner": ["-", "-"],
                 "head_index": 1}))
        self.assertEqual(
            "PLURAL",
            self.property_computer._compute_gender(
                {"tokens": ["the", "guys"], "head": ["guys"],
                 "pos": ["DT", "NNS"],"type": "NOM", "ner": ["-", "-"],
                 "head_index": 1}))
        self.assertEqual(
            "NEUTRAL",
            self.property_computer._compute_gender(
                {"tokens": ["the", "Mona", "Lisa"], "head": ["Mona", "Lisa"],
                 "pos": ["DT", "NNP", "NNP"],"type": "NAM",
                 "ner": ["-", "WORK_OF_ART", "WORK_OF_ART"], "head_index": 2}))

    def test_semantic_class(self):
        self.assertEqual(
            "PERSON",
            self.property_computer._compute_semantic_class(
                {"tokens": ["him"], "pos": ["PRP"], "type": "PRO",
                 "citation_form": "he", "head_index": 0}))
        self.assertEqual(
            "OBJECT",
            self.property_computer._compute_semantic_class(
                {"tokens": ["its"], "pos": ["PRP$"], "type": "PRO",
                 "citation_form": "it", "head_index": 0}))
        self.assertEqual(
            "PERSON",
            self.property_computer._compute_semantic_class(
                {"tokens": ["Mrs.", "Robinson"], "ner": ["PERSON", "PERSON"],
                 "pos": ["NNP", "NNP"],"type": "NAM", "head_index": 1}))
        self.assertEqual(
            "OBJECT",
            self.property_computer._compute_semantic_class(
                {"tokens": ["a", "house"], "head": ["house"],
                 "pos": ["DT", "NN"],"type": "NOM", "ner": ["NONE", "NONE"],
                 "head_index": 1}))
        self.assertEqual(
            "UNKNOWN",
            self.property_computer._compute_semantic_class(
                {"tokens": ["adsfg"], "head": ["adsfg"],
                 "pos": ["NN"],"type": "NOM", "ner": ["NONE", "NONE"],
                 "head_index": 0}))

    def test_citation_form(self):
        self.assertEqual(
            "they",
            self.property_computer._get_citation_form({"tokens": ["them"]}))
        self.assertEqual(
            "she",
            self.property_computer._get_citation_form({"tokens": ["her"]}))
        self.assertEqual(
            None,
            self.property_computer._get_citation_form({"tokens": ["why"]}))

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
        real_document.enrich_with_parse_trees()

        expected = 0
        head = nltk.ParentedTree.fromstring("(WHNP (WP who))")
        mention_subtree = self.property_computer.get_relevant_subtree(
            Span(29, 34), real_document)
        self.assertEqual(expected, self.property_computer.get_head_index(
            head, mention_subtree))

    def test_tree_is_apposition(self):
        self.assertEqual(
            True,
            self.property_computer._is_apposition({
                "parse_tree": nltk.ParentedTree.fromstring(
                    "(NP (NP (NP (NNP Secretary)) (PP (IN of) (NP "
                    "(NNP State)))) (NP (NNP Madeleine) (NNP Albright)))")}))
        self.assertEqual(
            False,
            self.property_computer._is_apposition({
                "parse_tree": nltk.ParentedTree.fromstring(
                    "(NP (NP (NNP Secretary)) (PP (IN of) "
                    "(NP (NNP State))))")}))
        self.assertEqual(
            False,
            self.property_computer._is_apposition({
                "parse_tree": nltk.ParentedTree.fromstring(
                    "(NP (NP (NP (NNP Al) (NNP Gore) (POS 's)) (NN campaign) "
                    "(NN manager)) (, ,) (NP (NNP Bill) (NNP Daley)) (, ,))")}))
        self.assertEqual(
            False,
            self.property_computer._is_apposition({
                "parse_tree": nltk.ParentedTree.fromstring(
                    "(NP (NP (NNS news)) (NP (CD today)))")}))
        self.assertEqual(
            False,
            self.property_computer._is_apposition({
                "parse_tree": nltk.ParentedTree.fromstring(
                    "(NP (NP (PRP$ his) (NN brother)) (, ,) (NP (PRP$ his) "
                    "(NN sister)))")}))
        self.assertEqual(
            True,
            self.property_computer._is_apposition({
                "parse_tree":  nltk.ParentedTree.fromstring(
                    "(NP (NP (NNP Barack) (NNP Obama)) (, ,) (NP (DT the) "
                    "(NN president)))")}))

    def test_mention_type(self):
        spans = [Span(37, 37),
                 Span(37, 37),
                 Span(11, 12),
                 Span(31, 32),
                 Span(8, 8),
                 Span(33, 34)]

        docs = [self.date_mention_document]*5 + [self.real_document]

        expected = ["NAM"]*5 + ["NOM"]

        for span, doc, exp in zip(spans, docs, expected):
            m = Mention.from_document(span, doc)
            self.property_computer.compute_properties(m, 1)

            self.assertEqual(exp, m.attributes["type"])

    def test_mention_get_head(self):
        spans = [Span(33, 34),
                 Span(3, 3),
                 Span(3, 3)]

        docs = [self.real_document, self.for_head_document, self.for_head_document]

        attrs = ["head", "head", "head_as_lowercase_string"]

        expected = [["massacre"], ["Wedding"], "wedding"]

        for span, doc, attr, exp in zip(spans, docs, attrs, expected):
            m = Mention.from_document(span, doc)
            self.property_computer.compute_properties(m, 1)

            self.assertEqual(exp, m.attributes[attr])

    def test_mention_get_governor(self):
        m = Mention.from_document(
                Span(0, 1),
                self.real_document)
        self.property_computer.compute_properties(m, 1)

        expected = "massacred"
        self.assertEqual(expected, m.attributes["governor"])

    def test_mention_get_ancestry(self):
        m = Mention.from_document(
                Span(11, 11),
                self.real_document)

        self.property_computer.compute_properties(m, 1)

        expected = "-L-VBN-L-NONE"
        self.assertEqual(
            expected,
            m.attributes["ancestry"])

        m = Mention.from_document(
            Span(0, 0),
            self.real_document)

        self.property_computer.compute_properties(m, 1)

        expected = "-R-NNS-R-VBN"
        self.assertEqual(
            expected,
            m.attributes["ancestry"])

    def test_mention_get_head_span(self):
        m = Mention.from_document(
            Span(8, 10),
            self.for_head_document)

        self.property_computer.compute_properties(m, 1)

        self.assertEqual(
            Span(9, 10),
            m.attributes["head_span"])

    def test_mention_get_fine_type(self):
        spans = [Span(33, 34), Span(21, 27), Span(22, 22), Span(45, 45)]
        docs = [self.real_document, self.date_mention_document, self.date_mention_document, self.date_mention_document]
        expected = ["DEF", "NOTDEF", "NOTDEF", "they"]

        for span, doc, exp in zip(spans, docs, expected):
            m = Mention.from_document(span, doc)
            self.property_computer.compute_properties(m, 1)

            self.assertEqual(exp, m.attributes["fine_type"])


if __name__ == '__main__':
    unittest.main()
