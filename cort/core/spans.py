""" Manage spans in documents. """

from cort.core import mixins


__author__ = 'smartschat'


class Span(mixins.ComparableMixin):
    """ Manage and compare spans in documents.

    Attributes:
        begin (int): The begin of the span.
        end (int): The end of the span (inclusive).
    """
    def __init__(self, begin, end):
        """ Initialize a span from a begin and an end position.

        Args:
            begin (int): The begin of the span.
            end (int): The end of the span.
        """
        self.begin = begin
        self.end = end

    def __str__(self):
        return "(" + str(self.begin) + ", " + str(self.end) + ")"

    def __repr__(self):
        return "(" + str(self.begin) + ", " + str(self.end) + ")"

    def __lt__(self, other):
        """ Check whether this span is less than another span.

        (a,b) < (c,d) if and only if a < c or a = c and b < d

        Args:
            other (Span): A span.

        Returns:
            True if this span is less than other, False otherwise.
        """
        if self.begin < other.begin:
            return True
        elif self.begin > other.begin:
            return False
        elif self.end < other.end:
            return True
        else:
            return False

    def embeds(self, other):
        """ Check whether this span embeds another span.

        Args:
            other (Span): A span.

        Returns:
            True if this span embeds other, False otherwise.
        """
        return self.begin <= other.begin and self.end >= other.end

    def __hash__(self):
        return hash((self.begin, self.end))

    @staticmethod
    def parse(span_string):
        """ Parse a string specification of a span to a Span object.

        Valid representations are for example "(1, 2)" or "(1,2)".

        Args:
            span_string (str): A string representation of a span.

        Returns:
            Span: The span corresponding to the string representation.
        """
        without_brackets = span_string.strip()[1:-1]
        splitted_and_stripped = [token.strip() for token
                                 in without_brackets.split(",")]
        return Span(
            int(splitted_and_stripped[0]),
            int(splitted_and_stripped[1]))
