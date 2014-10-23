from cort.core.external_data import GenderData

__author__ = 'smartschat'

import unittest


class TestGenderData(unittest.TestCase):
    def setUp(self):
        self.gender_data = GenderData.get_instance()

    def test_look_up(self):
        self.assertEqual("NEUTRAL",
                         self.gender_data.look_up({"tokens": ["snafu"]}))

        self.assertEqual("FEMALE",
                         self.gender_data.look_up(
                             {"tokens": ["Barbara", "Bush"],
                              "head": ["Barbara", "Bush"]}))

        self.assertEqual("MALE",
                         self.gender_data.look_up({
                             "tokens": ["Footballer", "Zidane"],
                             "head": ["Zidane"]}))

if __name__ == '__main__':
    unittest.main()
