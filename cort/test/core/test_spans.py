import unittest

from cort.core.spans import Span


__author__ = 'smartschat'


class TestSpan(unittest.TestCase):
    def test_span(self):
        span = Span(0, 1)
        self.assertEqual(0, span.begin)
        self.assertEqual(1, span.end)

    def test_parse(self):
        self.assertEqual(Span(10, 12), Span.parse("(10, 12)"))
        self.assertEqual(Span(10, 12), Span.parse("(10,12)"))

if __name__ == '__main__':
    unittest.main()
