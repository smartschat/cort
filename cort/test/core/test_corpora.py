import os
import unittest

from cort.core.corpora import Corpus


__author__ = 'smartschat'


class TestCorpora(unittest.TestCase):
    def setUp(self):
        directory = os.path.dirname(os.path.realpath(__file__)) + "/resources/"
        self.input_data = open(directory + "input.conll", "r")

    def test_conll_reader(self):
        corpus = Corpus.from_file("test", self.input_data)
        self.assertEqual(5, len(corpus.documents))

if __name__ == '__main__':
    unittest.main()
