import mention_property_computer


__author__ = 'smartschat'


class Mention:
    def __init__(self, document, span, attributes):
        self.document = document
        self.span = span
        self.attributes = attributes

    def __lt__(self, other):
        return self.span < other.span

    def get_attribute(self, attribute):
        return self.attributes[attribute]

    @staticmethod
    def from_document(span, document):
        attributes = {
            "tokens": document.tokens[span.begin:span.end + 1],
            "pos": document.pos[span.begin:span.end + 1],
            "ner": document.ner[span.begin:span.end + 1],
            "annotated_set_id": document.coref[span],
            "sentence_id": document.sentence_spans_to_id[document.get_embedding_sentence(span)],
            "parse_tree": mention_property_computer.get_relevant_subtree(span, document),
            "speaker": document.speakers[span.begin],
        }

        attributes["grammatical_function"] = mention_property_computer.get_grammatical_function(attributes["parse_tree"])

        attributes["head"], attributes["head_span"], attributes["head_index"] = mention_property_computer.compute_head(attributes["parse_tree"], span, attributes)

        attributes["type"] = mention_property_computer.get_type(attributes["pos"][attributes["head_index"]], attributes["ner"][attributes["head_index"]])
        attributes["fine_type"] = mention_property_computer.get_fine_type(attributes["type"], attributes["tokens"][0], attributes["pos"][0])

        if attributes["type"] == "PRO":
            attributes["citation_form"] = mention_property_computer.get_citation_form(attributes["tokens"][0])

        attributes["number"] = mention_property_computer.compute_number(attributes)
        attributes["gender"] = mention_property_computer.compute_gender(attributes)

        attributes["semantic_class"] = mention_property_computer.compute_semantic_class(attributes)

        attributes["is_apposition"] = mention_property_computer.tree_is_apposition(attributes["parse_tree"])

        return Mention(document, span, attributes)

    def __eq__(self, other):
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
        return repr(self.document) + ", " + str(self.span) + ": " + " ".join(self.attributes["tokens"])

    def __repr__(self):
        return repr(self.document) + ", " + str(self.span) + ": " + str(self.attributes["tokens"])

    def get_context(self, window):
        if window < 0 and self.span.begin + window >= 0:
            return self.document.tokens[self.span.begin + window:self.span.begin]
        elif window > 0 and self.span.end + window + 1 <= len(self.document.tokens):
            return self.document.tokens[self.span.end + 1:self.span.end + window + 1]