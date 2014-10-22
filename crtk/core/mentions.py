""" Manage mentions and their attributes. """

from crtk.core import mention_property_computer
from crtk.core import spans


__author__ = 'smartschat'


class Mention:
    """
    A mention is an expression in a document which is potentially referring.

    Attributes:
        document (CoNLLDocument): The document the mention belongs to.
        span (Span): The span of the mention in its document. If for example
            the span is (3, 4), then the mention starts at the 3rd token in
            the document and ends at the 4th (inclusive).
        attributes (dict(str, object)): A mapping of attribute names to
            attribute values. When creating a document from a text, The
            following attributes are used:
                tokens (list(str)): the tokens of the mention,
                head (list(str)): the head words of the mention,
                pos (list(str)): the part-of-speech tags of the mention
                ner (list(str)): the named entity tags of the mention,
                    as found in the data,
                type (str): the mention type, one of
                    NAM (proper name),
                    NOM (common noun(,
                    PRO (pronoun),
                    DEM (demonstrative pronoun),
                    VRB (verb),
                fine_type (str): only set when the mention is a nominal or a
                    pronoun, for nominals values
                        DEF (definite noun phrase) or
                        INDEF,
                    for pronouns values
                        PERS_NOM (personal pronoun, nominative case),
                        PERS_ACC (personal pronoun, accusative),
                        REFL (reflexive pronoun),
                        POSS (possessive pronoun) or
                        POSS_ADJ (possessive adjective, e.g. 'his'),
                citation_form (str): only set if the mention is a pronoun,
                    then the canonical form of the pronoun, i.e. one of
                    i, you, he, she, it, we, they,
                grammatical_function (str): either SUBJECT, OBJECT or OTHER,
                number (str): either SINGULAR, PLURAL or UNKNOWN,
                gender (str): either MALE, FEMALE, NEUTRAL, PLURAL or UNKNOWN,
                semantic_class (str): either PERSON, OBJECT or UNKNOWN,
                sentence_id (int): the sentence id of the mention's sentence
                    (starting at 0),
                parse_tree (nltk.ParentedTree): the parse tree of the mention,
                speaker (str): the speaker of the mention,
                antecedent (Mention): the antecedent of the mention
                    (intially None),
                annotated_set_id (str): the set id of the mention as found
                    in the data
                set_id (str): the set id of the mention computed by a
                    coreference resolution approach (initially None),
                head_span (Span): the span of the head (in the document),
                head_index (int): the mention-internal index of the start of
                    the head,
                is_apposition (bool): whether the mention contains an
                    apposition
    """
    def __init__(self, document, span, attributes):
        """
        Initialize a mention in a document.

        Args:
            document (CoNLLDocument): The document the mention belongs to.
            span (Span): The span of the mention in its document.
            attributes (dict(str, object)): A mapping of attribute names to
                attribute values (see the class documentation for more
                information).
        """
        self.document = document
        self.span = span
        self.attributes = attributes

    @staticmethod
    def from_document(span, document):
        """
        Create a mention from a span in a document.

        All attributes of the mention are computed from the linguistic
        information found in the document. For information about the
        attributes, see the class documentation.

        Args:
            document (CoNLLDocument): The document the mention belongs to.
            span (Span): The span of the mention in the document.

        Returns:
            (Mention): A mention extracted from the input span in the input
                document.
        """
        attributes = {
            "tokens": document.tokens[span.begin:span.end + 1],
            "pos": document.pos[span.begin:span.end + 1],
            "ner": document.ner[span.begin:span.end + 1],
            "annotated_set_id": document.coref[span],
            "sentence_id": document.sentence_spans_to_id[
                document.get_embedding_sentence(span)],
            "parse_tree": mention_property_computer.get_relevant_subtree(
                span, document),
            "speaker": document.speakers[span.begin],
            "antecedent": None,
            "set_id": None
        }

        attributes["is_apposition"] = \
            mention_property_computer.is_apposition(attributes)

        attributes["grammatical_function"] = \
            mention_property_computer.get_grammatical_function(attributes)

        (head, in_mention_span, head_index) = \
            mention_property_computer.compute_head_information(attributes)

        attributes["head"] = head
        attributes["head_span"] = spans.Span(
            span.begin + in_mention_span.begin,
            span.begin + in_mention_span.end
        )
        attributes["head_index"] = head_index

        attributes["type"] = mention_property_computer.get_type(attributes)
        attributes["fine_type"] = mention_property_computer.get_fine_type(
            attributes)

        if attributes["type"] == "PRO":
            attributes["citation_form"] = \
                mention_property_computer.get_citation_form(
                    attributes)

        attributes["number"] = \
            mention_property_computer.compute_number(attributes)
        attributes["gender"] = \
            mention_property_computer.compute_gender(attributes)

        attributes["semantic_class"] = \
            mention_property_computer.compute_semantic_class(attributes)

        return Mention(document, span, attributes)

    def __lt__(self, other):
        """ Check whether this mention is less than another mention.

        self < other if and only if self.span < other.span, that is,
            - when this mentions begins before the other mention, or
            - the mentions begin at the same position, but this mention ends
              before the other mention.

        Args:
            other (Mention): A mention.

        Returns:
            (bool): True if this mention is less than other, False otherwise.
        """
        return self.span < other.span

    def __eq__(self, other):
        """ Check for equality.

        Two mentions are equal if they are in the same document have the same
        span.

        Args:
            other (Mention): A mention.

        Returns:
            (bool): True if the mentions are in the same document and have the
                same span.
        """
        if isinstance(other, self.__class__):
            return self.span == other.span and self.document == other.document
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if self.document is None:
            return hash(self.span)
        else:
            return hash((self.document.id, self.document.part, self.span))

    def __str__(self):
        return (repr(self.document) +
                ", " +
                str(self.span) +
                ": "
                + " ".join(self.attributes["tokens"]))

    def __repr__(self):
        return (repr(self.document) +
                ", " +
                str(self.span) +
                ": " +
                str(self.attributes["tokens"]))

    def get_context(self, window):
        """ Get the context in a window around the mention.

        Args:
            window (int): An integer specifying the size of the window.

        Returns:
            (list(str)): The tokens in a window of |window| tokens to the
                right or the left, depending on the sign of window:
                    +: to the right,
                    -: to the left.
                Return None if the window is not contained in the document.
        """
        if window < 0 <= window + self.span.begin:
            return self.document.tokens[
                self.span.begin + window:self.span.begin]
        elif (window > 0 and self.span.end + window + 1
                <= len(self.document.tokens)):
            return self.document.tokens[
                self.span.end + 1:self.span.end + window + 1]
