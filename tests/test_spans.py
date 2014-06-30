from spans import Span
import unittest


__author__ = 'smartschat'


class TestSpan(unittest.TestCase):
    def test_span(self):
        span = Span(0, 1)
        self.assertEqual(0, span.begin)
        self.assertEqual(1, span.end)


if __name__ == '__main__':
    unittest.main()
