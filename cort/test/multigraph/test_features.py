import unittest

from cort.core.mentions import Mention
from cort.core.spans import Span
from cort.coreference.multigraph import features


__author__ = 'smartschat'


class TestFeatures(unittest.TestCase):
    def test_non_pronominal_string_match(self):
        self.assertEqual(
            True,
            features.non_pronominal_string_match(
                Mention(
                    None,
                    Span(0, 4),
                    {"tokens": ["the", "newly-elect", "leader", "'s", "wife"],
                     "pos": ["DT", "JJ", "NN", "POS", "NN"], "type": "NOM"}),
                Mention(
                    None,
                    Span(5, 7),
                    {"tokens": ["newly-elect", "leader", "wife"],
                     "pos": ["JJ", "NN", "NN"], "type": "NOM"})))
        self.assertEqual(
            False,
            features.non_pronominal_string_match(
                Mention(
                    None,
                    Span(0, 4),
                    {"tokens": ["the", "newly-elect", "leader", "'s", "wife"],
                     "pos": ["DT", "JJ", "NN", "POS", "NN"], "type": "NOM"}),
                Mention(
                    None,
                    Span(5, 5),
                    {"tokens": ["leader"], "pos": ["NN"], "type": "NOM"})))
        self.assertEqual(
            True,
            features.non_pronominal_string_match(
                Mention(
                    None,
                    Span(0, 0),
                    {"tokens": ["President"], "pos": ["NNP"], "type": "NAM"}),
                Mention(
                    None,
                    Span(1, 2),
                    {"tokens": ["the", "president"], "pos": ["DT", "NN"],
                     "type": "NOM"})))
        self.assertEqual(
            False,
            features.non_pronominal_string_match(
                Mention(
                    None,
                    Span(0, 0),
                    {"tokens": ["it"], "pos": ["PRP"], "type": "PRO"}),
                Mention(
                    None,
                    Span(1, 1),
                    {"tokens": ["IT"], "pos": ["NNP"], "type": "NAM"})))

    def test_head_match(self):
        self.assertEqual(
            True,
            features.head_match(
                Mention(
                    None,
                    Span(0, 4),
                    {"tokens": ["the", "newly-elect", "leader", "'s", "wife"],
                     "head": ["wife"], "type": "NOM",
                     "semantic_class": "PERSON"}),
                Mention(
                    None,
                    Span(5, 6),
                    {"tokens": ["the", "wife"], "head": ["wife"],
                     "type": "NOM", "semantic_class": "PERSON"})))
        self.assertEqual(
            False,
            features.head_match(
                Mention(
                    None,
                    Span(0, 4),
                    {"tokens": ["the", "newly-elect", "leader", "'s", "wife"],
                     "head": ["wife"], "type": "NOM",
                     "semantic_class": "PERSON"}),
                Mention(
                    None,
                    Span(5, 5),
                    {"tokens": ["leader"], "head": ["leader"], "type": "NOM",
                     "semantic_class": "PERSON"})))
        self.assertEqual(
            True,
            features.head_match(
                Mention(
                    None,
                    Span(0, 0),
                    {"tokens": ["President"], "head": ["President"],
                     "type": "NAM", "semantic_class": "PERSON"}),
                Mention(
                    None,
                    Span(1, 2),
                    {"tokens": ["the", "president"], "head": ["president"],
                     "type": "NOM", "semantic_class": "PERSON"})))
        self.assertEqual(
            False,
            features.head_match(
                Mention(
                    None,
                    Span(0, 0),
                    {"tokens": ["it"], "head": ["it"], "type": "PRO",
                     "semantic_class": "OBJECT"}),
                Mention(
                    None,
                    Span(1, 1),
                    {"tokens": ["it"], "head": ["it"], "type": "PRO",
                     "semantic_class": "OBJECT"})))
        self.assertEqual(
            False,
            features.head_match(
                Mention(
                    None,
                    Span(0, 1),
                    {"tokens": ["10", "percent"], "head": ["percent"],
                     "type": "NOM", "semantic_class": "NUMERIC"}),
                Mention(
                    None,
                    Span(2, 3),
                    {"tokens": ["Some", "percent"], "head": ["percent"],
                     "type": "NOM", "semantic_class": "NUMERIC"})))

    def test_pronoun_same_canonical_form(self):
        self.assertEqual(
            True,
            features.pronoun_same_canonical_form(
                Mention(
                    None,
                    Span(0, 0),
                    {"tokens": ["he"], "type": "PRO", "citation_form": "he"}),
                Mention(
                    None,
                    Span(1, 1),
                    {"tokens": ["he"], "type": "PRO", "citation_form": "he"})))

        self.assertEqual(
            True,
            features.pronoun_same_canonical_form(
                Mention(
                    None,
                    Span(0, 0),
                    {"tokens": ["he"], "type": "PRO", "citation_form": "he"}),
                Mention(
                    None,
                    Span(1, 1),
                    {"tokens": ["him"], "type": "PRO",
                     "citation_form": "he"})))

        self.assertEqual(
            False,
            features.pronoun_same_canonical_form(
                Mention(
                    None,
                    Span(0, 0),
                    {"tokens": ["US"], "type": "NAM"}),
                Mention(
                    None,
                    Span(1, 1),
                    {"tokens": ["us"], "type": "PRO"})))

    def test_speaker(self):
        self.assertEqual(True, features.speaker(
            Mention(None, Span(3, 3), {"tokens": ["you"], "type": "PRO", "citation_form": "you", "speaker": "snafu"}),
            Mention(None, Span(0, 0), {"tokens": ["I"], "type": "PRO", "citation_form": "i", "speaker": "foo"}),
        ))

        self.assertEqual(True, features.speaker(
            Mention(None, Span(3, 3), {"tokens": ["me"], "type": "PRO", "citation_form": "i", "speaker": "snafu"}),
            Mention(None, Span(0, 0), {"tokens": ["I"], "type": "PRO", "citation_form": "i", "speaker": "snafu"}),
        ))

        self.assertEqual(False, features.speaker(
            Mention(None, Span(3, 3), {"tokens": ["you"], "type": "PRO", "citation_form": "you", "speaker": "snafu"}),
            Mention(None, Span(0, 0), {"tokens": ["I"], "type": "PRO", "citation_form": "i", "speaker": "snafu"}),
        ))

        self.assertEqual(True, features.speaker(
            Mention(None, Span(3, 3), {"tokens": ["me"], "type": "PRO", "citation_form": "i", "speaker": "snafu"}),
            Mention(None, Span(0, 0), {"tokens": ["snafu"], "type": "NAM", "speaker": "-", "head": ["snafu"]}),
        ))

        self.assertEqual(False, features.speaker(
            Mention(None, Span(3, 3), {"tokens": ["me"], "type": "PRO", "citation_form": "i", "speaker": "snafu"}),
            Mention(None, Span(0, 0), {"tokens": ["foo"], "type": "NAM", "speaker": "-", "head": ["foo"]}),
        ))

    def test_not_pronoun_distance(self):
        self.assertEqual(False, features.not_pronoun_distance(
            Mention(None, Span(0, 0), {"tokens": ["he"], "type": "PRO", "citation_form": "he", "sentence_id": 10}),
            Mention(None, Span(100, 100), {"tokens": ["he"], "type": "PRO", "citation_form": "he", "sentence_id": 0})
        ))

        self.assertEqual(True, features.not_pronoun_distance(
            Mention(None, Span(100, 100), {"tokens": ["it"], "type": "PRO", "citation_form": "it", "sentence_id": 10}),
            Mention(None, Span(0, 0), {"tokens": ["company"], "type": "NOM", "sentence_id": 0})
        ))

        self.assertEqual(False, features.not_pronoun_distance(
            Mention(None, Span(100, 100), {"tokens": ["them"], "type": "PRO", "citation_form": "they", "sentence_id": 10}),
            Mention(None, Span(0, 0), {"tokens": ["company"], "type": "NOM", "sentence_id": 0})
        ))

        self.assertEqual(False, features.not_pronoun_distance(
            Mention(None, Span(100, 100), {"tokens": ["them"], "type": "PRO", "citation_form": "they", "sentence_id": 1}),
            Mention(None, Span(0, 0), {"tokens": ["company"], "type": "NOM", "sentence_id": 0})
        ))

    def test_not_speaker(self):
        self.assertEqual(True, features.not_speaker(
            Mention(None, Span(3, 3), {"tokens": ["I"], "type": "PRO", "citation_form": "i", "speaker": "snafu"}),
            Mention(None, Span(0, 0), {"tokens": ["I"], "type": "PRO", "citation_form": "i", "speaker": "foo"}),
        ))

        self.assertEqual(True, features.not_speaker(
            Mention(None, Span(3, 3), {"tokens": ["us"], "type": "PRO", "citation_form": "we", "speaker": "snafu"}),
            Mention(None, Span(0, 0), {"tokens": ["we"], "type": "PRO", "citation_form": "we", "speaker": "foo"}),
        ))

        self.assertEqual(False, features.not_speaker(
            Mention(None, Span(3, 3), {"tokens": ["you"], "type": "PRO", "citation_form": "you", "speaker": "snafu"}),
            Mention(None, Span(0, 0), {"tokens": ["I"], "type": "PRO", "citation_form": "i", "speaker": "foo"}),
        ))

        self.assertEqual(True, features.not_speaker(
            Mention(None, Span(3, 3), {"tokens": ["you"], "type": "PRO", "citation_form": "you", "speaker": "snafu"}),
            Mention(None, Span(0, 0), {"tokens": ["I"], "type": "PRO", "citation_form": "i", "speaker": "snafu"}),
        ))

    def test_not_embedding(self):
        self.assertEqual(True, features.not_embedding(
            Mention(None, Span(3, 3), {"tokens": ["it"], "type": "PRO", "fine_type": "PERS"}),
            Mention(None, Span(0, 4), {"tokens": ["the", "company", "which", "it", "bought"], "type": "NOM"})
        ))

        self.assertEqual(False, features.not_embedding(
            Mention(None, Span(3, 3), {"tokens": ["its"], "type": "PRO", "fine_type": "POSS_ADJ"}),
            Mention(None, Span(0, 4), {"tokens": ["the", "company", "which", "loves", "its", "success"], "type": "NOM"})
        ))

    def test_not_compatible(self):
        self.assertEqual(True, features.not_compatible(
            Mention(None, Span(0, 0), {"tokens": ["he"], "pos": ["PRP"], "type": "PRO", "number": "SINGULAR", "gender": "MALE", "semantic_class": "PERSON"}),
            Mention(None, Span(1, 1), {"tokens": ["she"], "pos": ["PRP"], "type": "PRO", "number": "SINGULAR", "gender": "FEMALE", "semantic_class": "PERSON"})
        ))

        self.assertEqual(False, features.not_compatible(
            Mention(None, Span(0, 0), {"tokens": ["he"], "pos": ["PRP"], "type": "PRO", "number": "SINGULAR", "gender": "MALE", "semantic_class": "PERSON"}),
            Mention(None, Span(1, 1), {"tokens": ["slawabu"], "pos": ["NN"], "type": "NOM", "number": "UNKNOWN", "gender": "UNKNOWN", "semantic_class": "PERSON"})
        ))

        self.assertEqual(False, features.not_compatible(
            Mention(None, Span(0, 0), {"tokens": ["Jesus"], "pos": ["NNP"], "type": "NAM", "number": "SINGULAR", "gender": "MALE", "semantic_class": "PERSON"}),
            Mention(None, Span(1, 1), {"tokens": ["Jesus"], "pos": ["NNP"], "type": "NAM", "number": "SINGULAR", "gender": "UNKNOWN", "semantic_class": "NORP"})
        ))

    def test_alias(self):
        self.assertEqual(False, features.alias(
            Mention(None, Span(0, 0), {"tokens": ["he"], "type": "PRO", "citation_form": "he"}),
            Mention(None, Span(1, 1), {"tokens": ["he"], "type": "PRO", "citation_form": "he"})
        ))

        self.assertEqual(False, features.alias(
            Mention(None, Span(0, 2), {"head": ["International", "Business", "Machines"], "type": "NAM", "ner": ["ORG", "ORG", "ORG"], "head_index": 2}),
            Mention(None, Span(3, 5), {"head": ["International", "Business", "Machines"], "type": "NAM", "ner": ["ORG", "ORG", "ORG"], "head_index": 2})
        ))

        self.assertEqual(True, features.alias(
            Mention(None, Span(0, 2), {"head": ["International", "Business", "Machines"], "type": "NAM", "ner": ["ORG", "ORG", "ORG"], "head_index": 2}),
            Mention(None, Span(3, 3), {"head": ["IBM"], "type": "NAM", "ner": ["ORG"], "head_index": 0})
        ))

    def test_not_modifier(self):
        self.assertEqual(
            True,
            features.not_modifier(
                Mention(None, Span(10, 16), {"tokens": ["the", "long-awaited", "beginning", "of", "a", "new", "century"], "type": "NOM", "head_span": Span(12, 12), "pos": ["DT", "JJ", "NN", "IN", "DT", "JJ", "NN"]}),
                Mention(None, Span(0, 1), {"tokens": ["the", "beginning"], "type": "NOM", "head_span": Span(1, 1), "pos": ["DT", "NN"]})
            )
        )

        self.assertEqual(
            False,
            features.not_modifier(
                Mention(None, Span(18, 19), {"tokens": ["the", "beginning"], "type": "NOM", "head_span": Span(19, 19), "pos": ["DT", "NN"]}),
                Mention(None, Span(10, 16), {"tokens": ["the", "long-awaited", "beginning", "of", "a", "new", "century"], "type": "NOM", "head_span": Span(12, 12), "pos": ["DT", "JJ", "NN", "IN", "DT", "JJ", "NN"]}),
            )
        )

        self.assertEqual(
            False,
            features.not_modifier(
                Mention(None, Span(18, 19), {"tokens": ["cool", "people"], "type": "NOM", "head_span": Span(19, 19), "pos": ["JJ", "NNS"]}),
                Mention(None, Span(10, 11), {"tokens": ["Cool", "people"], "type": "NOM", "head_span": Span(11, 11), "pos": ["JJ", "NNS"]}),
            )
        )

    def test_get_modifiers(self):
        self.assertEqual(
            set(["long-awaited", "new", "century"]),
            features.get_modifier(Mention(None, Span(10, 16), {"tokens": ["the", "long-awaited", "beginning", "of", "a", "new", "century"], "head_span": Span(12, 12), "pos": ["DT", "JJ", "NN", "IN", "DT", "JJ", "NN"]}))
        )

    def test_get_category_for_alias(self):
        self.assertEqual("PERSON", features.get_category_for_alias("PERSON", "PERSON"))
        self.assertEqual(None, features.get_category_for_alias("LOC", "GPE"))
        self.assertEqual("LOC", features.get_category_for_alias("LOC", "LOC"))
        self.assertEqual(None, features.get_category_for_alias("WORK_OF_ART", "WORK_OF_ART"))

    def test_loc_alias(self):
        self.assertEqual(True, features.loc_alias(["Washington"], ["Washington", ",", "DC"]))
        self.assertEqual(True, features.loc_alias(["NJ"], ["New", "Jersey"]))

    def test_org_alias(self):
        self.assertEqual(True, features.org_alias(["IBM"], ["International", "Business", "Machines"]))
        self.assertEqual(True, features.org_alias(["Lomas", "Corp"], ["Lomas"]))

    def test_person_alias(self):
        self.assertEqual(True, features.person_alias(["Bill", "Clinton"], ["Clinton"]))
        self.assertEqual(True, features.person_alias(["Hillary", "Clinton"], ["Hillary"]))
        self.assertEqual(True, features.person_alias(["Mr.", "Clinton"], ["Bill", "Clinton"]))
        self.assertEqual(False, features.person_alias(["Mr.", "Bush"], ["Bill", "Clinton"]))
        self.assertEqual(False, features.person_alias(["President", "Barack", "Obama"], ["US", "President", "Obama"]))
        self.assertEqual(True, features.person_alias(["President", "Barack", "Obama"], ["President", "Obama"]))
        self.assertEqual(True, features.person_alias(["VOA", "'s", "Barry", "Wood"], ["Barry", "Wood"]))

    def test_starts_with(self):
        self.assertEqual(True, features.starts_with(["Washington"], ["Washington", ",", "DC"]))
        self.assertEqual(True, features.starts_with(["Lomas", "Corp"], ["Lomas"]))
        self.assertEqual(False, features.starts_with(["Lomas", "Corp"], ["Washington"]))

    def test_is_abbreviation(self):
        self.assertEqual(True, features.is_abbreviation(["IBM"], ["I.B.M."]))
        self.assertEqual(True, features.is_abbreviation(["IBM"], ["International", "Business", "Machines"]))
        self.assertEqual(True, features.is_abbreviation(["Darkhorse", "Productions", "Inc."], ["DP"]))
        self.assertEqual(False, features.is_abbreviation(["Darkhorse", "Productions", "Inc."], ["IBM"]))

    def test_get_acronyms(self):
        self.assertEqual(("International Business Machines", "IBM", "I.B.M."),
                         features.get_acronyms(["International", "Business", "Machines"]))

        self.assertEqual(("Darkhorse Productions", "DP", "D.P."),
                         features.get_acronyms(["Darkhorse", "Productions", "Inc."]))

        self.assertEqual(("Darkhorse Productions", "DP", "D.P."),
                         features.get_acronyms(["Darkhorse", "Productions", "Inc"]))



if __name__ == '__main__':
    unittest.main()
