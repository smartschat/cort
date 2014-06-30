from mixin import ComparableMixin


__author__ = 'smartschat'


class Span(ComparableMixin):
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end

    def __str__(self):
        return "(" + str(self.begin) + ", " + str(self.end) + ")"

    def __repr__(self):
        return "(" + str(self.begin) + ", " + str(self.end) + ")"

    def __lt__(self, other):
        if self.begin < other.begin:
            return True
        elif self.begin > other.begin:
            return False
        elif self.end < other.end:
            return True
        else:
            return False

    def embeds(self, other):
        return self.begin <= other.begin and self.end >= other.end

    def __hash__(self):
        return hash((self.begin, self.end))