from cort.core.util import clean_via_pos

__author__ = 'smartschat'

import unittest


class TestUtil(unittest.TestCase):
    def test_clean_via_pos(self):
        self.assertEqual(
            ["newly-elect", "leader", "wife"],
            clean_via_pos(
                ["the", "newly-elect", "leader", "'s", "wife"],
                ["DT", "JJ", "NN", "POS", "NN"]))


if __name__ == '__main__':
    unittest.main()
