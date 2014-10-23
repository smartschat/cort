""" Mixins. """


__author__ = 'smartschat'


class ComparableMixin:
    """ A mixin for deducing comparison operators from __lt__. """
    def __eq__(self, other):
        if self is None and other is not None:
            return False
        elif self is not None and other is None:
            return False
        else:
            return not self < other and not other < self

    def __ne__(self, other):
        return self < other or other < self

    def __gt__(self, other):
        return other < self

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not other < self
