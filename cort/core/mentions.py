""" Manage mentions and their attributes. """

from cort.core import mention_property_computer
from cort.core import spans


__author__ = 'smartschat'


class Mention:
    """ A mention is an expression in a document which is potentially referring.

    Attributes:
        document (CoNLLDocument): The document the mention belongs to.
        span (Span): The span of the mention in its document. If for example
            the span is (3, 4), then the mention starts at the 3rd token in
            the document and ends at the 4th (inclusive).
        attributes (dict(str, object)): A mapping of attribute names to
            attribute values. When creating a document from a text, The
            following attributes are used:

                - tokens (list(str)): the tokens of the mention,
                - head (list(str)): the head words of the mention,
                - pos (list(str)): the part-of-speech tags of the mention,
                - ner (list(str)): the named entity tags of the mention,
                  as found in the data,
                - type (str): the mention type, one of

                    - NAM (proper name),
                    - NOM (common noun),
                    - PRO (pronoun),
                    - DEM (demonstrative pronoun),
                    - VRB (verb),

                - fine_type (str): only set when the mention is a nominal or a
                  pronoun, for nominals values

                    - DEF (definite noun phrase) or
                    - INDEF,

                  for pronouns values

                    - PERS_NOM (personal pronoun, nominative case),
                    - PERS_ACC (personal pronoun, accusative),
                    - REFL (reflexive pronoun),
                    - POSS (possessive pronoun) or
                    - POSS_ADJ (possessive adjective, e.g. 'his'),

                - citation_form (str): only set if the mention is a pronoun,
                  then the canonical form of the pronoun, i.e. one of
                  i, you, he, she, it, we, they,
                - grammatical_function (str): either SUBJECT, OBJECT or OTHER,
                - number (str): either SINGULAR, PLURAL or UNKNOWN,
                - gender (str): either MALE, FEMALE, NEUTRAL, PLURAL or UNKNOWN,
                - semantic_class (str): either PERSON, OBJECT or UNKNOWN,
                - sentence_id (int): the sentence id of the mention's sentence
                  (starting at 0),
                - parse_tree (nltk.ParentedTree): the parse tree of the mention,
                - speaker (str): the speaker of the mention,
                - antecedent (Mention): the antecedent of the mention
                  (intially None),
                - annotated_set_id (str): the set id of the mention as found
                  in the data,
                - set_id (str): the set id of the mention computed by a
                  coreference resolution approach (initially None),
                - head_span (Span): the span of the head (in the document),
                - head_index (int): the mention-internal index of the start of
                  the head,
                - is_apposition (bool): whether the mention contains an
                  apposition,
                - head_as_lowercase_string (str): the head lowercased and as a
                  string,
                - tokens_as_lowercase_string (str): all tokens of the mention
                  lowercased and as a string,
                - first_in_gold_entity (bool): whether the mention is the first
                  mention in its gold entity (for system mentions, this is
                  also true if no preceding mention in the same entity was
                  found by the mention extractor),

    """
    def __init__(self, document, span, attributes):
        """ Initialize a mention in a document.

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
    def dummy_from_document(document):

        return Mention(document, None, {
            "is_dummy": True,
            "annotated_set_id": None,
            "tokens": [],
            "first_in_gold_entity": True
        })

    def is_dummy(self):
        return "is_dummy" in self.attributes and self.attributes["is_dummy"]

    @staticmethod
    def from_document(span, document, first_in_gold_entity=False):
        """
        Create a mention from a span in a document.

        All attributes of the mention are computed from the linguistic
        information found in the document. For information about the
        attributes, see the class documentation.

        Args:
            document (CoNLLDocument): The document the mention belongs to.
            span (Span): The span of the mention in the document.

        Returns:
            Mention: A mention extracted from the input span in the input
            document.
        """

        i, sentence_span = document.get_sentence_id_and_span(span)

        attributes = {
            "tokens": document.tokens[span.begin:span.end + 1],
            "pos": document.pos[span.begin:span.end + 1],
            "ner": document.ner[span.begin:span.end + 1],
            "sentence_id": i,
            "parse_tree": mention_property_computer.get_relevant_subtree(
                span, document),
            "speaker": document.speakers[span.begin],
            "antecedent": None,
            "set_id": None,
            "first_in_gold_entity": first_in_gold_entity
        }

        if span in document.coref:
            attributes["annotated_set_id"] = document.coref[span]
        else:
            attributes["annotated_set_id"] = None

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

        attributes["head_as_lowercase_string"] = " ".join(attributes[
            "head"]).lower()

        attributes["tokens_as_lowercase_string"] = " ".join(attributes[
            "tokens"]).lower()

        dep_tree = document.dep[i]

        index = span.begin + head_index - sentence_span.begin

        governor_id = dep_tree[index].head - 1

        if governor_id == -1:
            attributes["governor"] = "NONE"
        else:
            attributes["governor"] = dep_tree[governor_id].form.lower()

        attributes["ancestry"] = Mention._get_ancestry(dep_tree, index)

        attributes["deprel"] = dep_tree[index].deprel

        return Mention(document, span, attributes)

    @staticmethod
    def _get_ancestry(dep_tree, index, level=0):
        if level >= 2:
            return ""
        else:
            governor_id = dep_tree[index].head - 1

            direction = "L"

            if governor_id > index:
                direction = "R"

            if governor_id == -1:
                return  "-" + direction + "-NONE"
            else:
                return "-" + direction + "-" + dep_tree[governor_id].pos + \
                    Mention._get_ancestry(dep_tree, governor_id, level+1)



    def __lt__(self, other):
        """ Check whether this mention is less than another mention.

        ``self < other`` if and only if ``self.span < other.span``, that is,

            - this mention has span None (is a dummy mention), and the other
              mention has a span which is not None, or
            - this mentions begins before the other mention, or
            - the mentions begin at the same position, but this mention ends
              before the other mention.

        Args:
            other (Mention): A mention.

        Returns:
            True if this mention is less than other, False otherwise.
        """

        if self.span is None:
            return other.span is not None
        elif other.span is None:
            return False
        else:
            return self.span < other.span

    def __eq__(self, other):
        """ Check for equality.

        Two mentions are equal if they are in the same document and have the
        same span.

        Args:
            other (Mention): A mention.

        Returns:
            True if the mentions are in the same document and have the same
            span.
        """
        if isinstance(other, self.__class__):
            return self.span == other.span and self.document == other.document
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if self.document is None:
            return hash((self.span.begin, self.span.end))
        elif self.span is None:
            return hash(self.document.identifier)
        else:
            return hash((self.document.identifier,
                         self.span.begin,
                         self.span.end))

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
            list(str): The tokens in a window of around the mention.

            In particular, get ``window`` tokens to the right or left of the
            mention,, depending on the sign of ``window``: if the sign is +,
            then to the right, if the sign is -, then to the left. Return
            None if the window is not contained in the document.
        """
        if window < 0 <= window + self.span.begin:
            return self.document.tokens[
                self.span.begin + window:self.span.begin]
        elif (window > 0 and self.span.end + window + 1
                <= len(self.document.tokens)):
            return self.document.tokens[
                self.span.end + 1:self.span.end + window + 1]

    def is_coreferent_with(self, m):
        """ Return whether this mention is coreferent with another mention.

        Args:
            m (Mention): Another mention.

        Returns:
            True if m and this mention are coreferent (are in the same document
            and have the same annotated set id), False otherwise.
        """

        self_set_id = self.attributes['annotated_set_id']
        m_set_id = m.attributes['annotated_set_id']

        if self.document is None and m.document is None:
            return self_set_id is not None and self_set_id == m_set_id
        elif self.is_dummy():
            return m.is_dummy()
        elif m.is_dummy():
            return self.is_dummy()
        else:
            return self.document == m.document \
                and self_set_id is not None \
                and self_set_id == m_set_id

    def decision_is_consistent(self, m):
        """ Return whether the decision to put this mention and m into the
        same entity is consistent with the gold annotation.

        The decision is consistent if one of the following conditions holds:

            - the mentions are coreferent,
            - one of the mentions is the dummy mention, and the other mention
              does not have a preceding mention that it is coreferent with.

        Args:
            m (Mention): Another mention.

        Returns:
            True if m and this mention are consistent according to the
            definition above, False otherwise.
        """

        if self.is_coreferent_with(m):
            return True
        elif self.is_dummy():
            return m.attributes['annotated_set_id'] is None \
                   or m.attributes["first_in_gold_entity"]
        elif m.is_dummy():
            return self.attributes['annotated_set_id'] is None \
                   or self.attributes["first_in_gold_entity"]
        else:
            return False
