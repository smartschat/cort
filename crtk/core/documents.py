""" Manage individual documents of text collections.

In particular, classes in this module allow to access the linguistic
information present in documents.
"""

from collections import defaultdict
import logging
import re

from crtk.core import mentions
from crtk.core import nltk_util
from crtk.core import spans


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__author__ = 'smartschat'


class CoNLLDocument:
    """Represents a document in CoNLL format.

    For a specification of the format, see
        http://conll.cemantix.org/2012/data.html.

    Attributes:
        folder: do we really need this?
        id (str): The id of the document,
        part (str): The part number of the document.
        genre (str): The genre the document belongs to.
        document_table (list(list(str))): A tabular representation of the
            document (as in the CoNLL data).
        in_sentence_ids (list(int)): All sentence ids in the document.
        indexing_start (int): The first sentence id in the document.
        tokens (list(str)): All tokens.
        pos (list(str)): All part-of-speech tags.
        ner (list(str)): All named entity tags (if a token does not have a
            tag, the tag is set to NONE).
        parse (list(str): All parse trees (in string list representation, as
            in the ConLL data).
        speakers (list(str)) = All speaker ids,
        sentence_spans_to_id (dict(Span, int)): A mapping of sentence spans to
            sentence ids.
        coref (defaultdict(span, int)): A mapping of mention spans to their
            coreference set id.
        spans_to_annotated_mentions (dict(Span, Mention)): A mapping of
            mention spans of the mentions annotated in the document to the
            corresponding mentions.
        annotated_mentions list(Mention): All annotated mentions.
        system_mentions list(Mention): The system mentions (initially empty).
        antecedent_decisions dict(Span, Span): Maps anaphor to antecedent
            (initially empty).
    """
    def __init__(self, document_as_string):
        """ Construct a document from a string representation.

        The Format must follow the CoNLL format, see
            http://conll.cemantix.org/2012/data.html.

        Args:
            document_as_string (str): A representation of a document in
                the CoNLL format.
        """
        begin = document_as_string.split("\n")[0]

        self.folder = "/".join(begin.split()[2].split("/")[0:-1])[1:] + "/"
        self.id = begin.split()[2].split("/")[-1][0:-2]
        self.part = begin.split()[-1]
        self.genre = self.__get_genre()

        self.document_table = CoNLLDocument.__string_to_table(
            document_as_string)

        self.in_sentence_ids = [int(i) for i in self.__extract_from_column(2)]
        # if in_sentence_ids are 1-based, fix this
        self.indexing_start = self.in_sentence_ids[0]
        if self.indexing_start != 0:
            logger.warning("Detected " +
                           str(self.indexing_start) +
                           "-based indexing for tokens in sentences in input,"
                           "transformed to 0-based indexing.")
            self.in_sentence_ids = [i - self.indexing_start
                                    for i in self.in_sentence_ids]

        self.tokens = self.__extract_from_column(3)
        self.pos = self.__extract_from_column(4)
        self.ner = self.__extract_ner()
        self.parse = self.__extract_from_column(5)
        self.speakers = self.__extract_from_column(9)
        self.sentence_spans_to_id = self.__extract_sentence_spans()

        self.sentence_spans_to_parses = {}
        for span in self.sentence_spans_to_id:
            parse = self.get_parse(span)
            tree = nltk_util.parse_parented_tree(parse)
            self.sentence_spans_to_parses[span] = tree

        self.coref = CoNLLDocument.__get_span_to_id(
            self.__extract_from_column(-1))

        # maps spans to mention objects
        self.spans_to_annotated_mentions = \
            self.__get_span_to_annotated_mentions()
        self.annotated_mentions = sorted(
            list(self.spans_to_annotated_mentions.values()))

        self.system_mentions = []

        self.antecedent_decisions = {}

    def __repr__(self):
        return self.id + ", part " + self.part

    def __hash__(self):
        return hash((self.id, self.part))

    def __lt__(self, other):
        """ Check whether this document is less than another document.

        It is less than another document if and only if
            (self.id, self.part) < (another.id, another.part),
        where the ids are compared lexicographically.

        Args:
            other (ConLLDocument): A document.

        Returns:
            (bool): True if this document is less than other, False otherwise.
        """
        return (self.id, self.part) < (other.id, other.part)

    def __eq__(self, other):
        """ Check for document equality.

        Two documents are considered equal is they have the same id and the
        same part.

        Args:
            other (ConLLDocument): A document.

        Returns:
            (bool): True if id and part of the documents are equal.
        """
        if isinstance(other, self.__class__):
            return self.id == other.id and self.part == other.part
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __get_genre(self):
        if re.match("^(a2e|eng)", self.id):
            return "wb"
        elif re.match("^(cctv|cnn_000|msnbc|phoenix)", self.id):
            return "bc"
        elif re.match("^(abc|c2e|cnn|mnb|nbc|pri|voa)", self.id):
            return "bn"
        elif re.match("^(chtb|wsj)", self.id):
            return "nw"
        elif re.match("^(ch)", self.id):
            return "tc"
        elif re.match("^(ectb)", self.id):
            return "mz"
        elif re.match("^(nt)", self.id):
            return "pt"
        else:
            return "unknown"

    def __extract_from_column(self, column):
        entries = []
        for line in self.document_table:
            entries.append(line[column])

        return entries

    def __extract_ner(self):
        entries = self.__extract_from_column(10)
        ner = []

        tag = "NONE"
        for i in range(0, len(entries)):
            entry = entries[i]

            if "(" in entry:
                tag = entry.strip("(").strip(")").strip("*")

            ner.append(tag)

            if ")" in entry:
                tag = "NONE"

        return ner

    @staticmethod
    def __string_to_table(document_as_string):
        table = []

        document_contents = document_as_string.split("\n")[1:-2]

        for line in document_contents:
            if line != "" and not line.isspace():
                table.append(line.split())

        return table

    def __extract_sentence_spans(self):
        sentence_spans_to_id = {}
        sentence_id = 0

        span_start = 0

        for i in range(1, len(self.in_sentence_ids)):
            if self.in_sentence_ids[i] <= self.in_sentence_ids[i-1]:
                sentence_spans_to_id[spans.Span(span_start, i-1)] = sentence_id
                sentence_id += 1
                span_start = i

        sentence_spans_to_id[
            spans.Span(span_start, len(self.in_sentence_ids)-1)] = sentence_id

        return sentence_spans_to_id

    @staticmethod
    def __get_span_to_id(column):
        span_to_id = defaultdict(int)

        ids_to_stack = defaultdict(list)

        for i in range(0, len(column)):
            entry = column[i]

            if entry != "-":
                parallel_annotations = entry.split("|")

                for annotation in parallel_annotations:
                    if annotation.startswith("(") and annotation.endswith(")"):
                        set_id = annotation[1:-1]
                        span_to_id[spans.Span(i, i)] = int(set_id)
                    elif annotation.startswith("("):
                        set_id = annotation[1:]
                        ids_to_stack[set_id].append(i)
                    elif annotation.endswith(")"):
                        set_id = annotation[:-1]
                        span_to_id[
                            spans.Span(ids_to_stack[set_id].pop(), i)
                        ] = int(set_id)

        return span_to_id

    def __get_span_to_annotated_mentions(self):
        mention_spans = self.coref.keys()

        mention_spans = sorted(mention_spans)
        span_to_mentions = {}

        for span in mention_spans:
            span_to_mentions[span] = mentions.Mention.from_document(span, self)

        return span_to_mentions

    def get_parse(self, span):
        """ Get a the parse tree (as a string) of to the span.

        Args:
            span (Span): A span corresponding to a fragment of the document.

        Returns:
            (str): A string representation of the parse tree of
                the span.
        """
        parse_tree = ""
        for i in range(span.begin, span.end+1):
            parse_bit = self.parse[i]
            parse_tree += \
                parse_bit.replace("(", " (").replace(
                    "*", " (" + self.pos[i] + " " + self.tokens[i] + ")")

        return parse_tree.strip()

    def get_embedding_sentence(self, span):
        """ Get the sentence span from the sentence embedding the span.

        Args:
            span (Span): A span corresponding to a fragment of the document.

        Returns:
            (Span): The span of the sentence which embeds the
                text corresponding to the span.
        """
        for sentence_span in self.sentence_spans_to_id.keys():
            if sentence_span.embeds(span):
                return sentence_span

    def are_coreferent(self, m, n):
        """ Return whether to mentions are coreferent.

        Args:
            m (Mention): A mention.
            n (Mention): Another mention.

        Returns:
            True if m and n are coreferent (are in the same document and have
            the same annotated set id), False otherwise.
        """
        return (m.document == n.document
                and m.document == self
                and m.span in self.spans_to_annotated_mentions
                and n.span in self.spans_to_annotated_mentions
                and self.spans_to_annotated_mentions[m.span].attributes[
                    "annotated_set_id"] == self.spans_to_annotated_mentions[
                        n.span].attributes["annotated_set_id"])

    def get_string_representation(self):
        """ Get a string representation of the document.

        Returns:
            (str): A string representation of the document which conforms to
                the CoNLL format specifications
                (http://conll.cemantix.org/2012/data.html).
        """
        mention_string_representation = \
            CoNLLDocument.__get_string_representation_of_mentions(
                len(self.document_table), self.system_mentions)

        new_table = self.document_table

        for row, mention_row in zip(new_table, mention_string_representation):
            row[-1] = mention_row

        padded_table = []

        current_row = new_table[0]
        padded_table.append(current_row)

        for i in range(1, len(new_table)):
            previous_row = current_row
            current_row = new_table[i]

            if int(current_row[2]) <= int(previous_row[2]):
                padded_table.append("")

            padded_table.append(current_row)

        begin = ("#begin document (" +
                 self.folder +
                 self.id +
                 "); part " +
                 self.part +
                 "\n")

        content = "\n".join(["\t".join(row) for row in padded_table])

        end = "\n#end document\n"

        return begin + content + end

    @staticmethod
    def __get_string_representation_of_mentions(length, mentions_in_doc):
        index_to_strings = defaultdict(list)

        for mention in mentions_in_doc:
            set_id = mention.attributes["set_id"]

            if set_id is None:
                continue

            span = mention.span

            if span.begin == span.end:
                index_to_strings[span.begin].append("(" + str(set_id) + ")")
            else:
                index_to_strings[span.begin].append("(" + str(set_id))
                index_to_strings[span.end].append(str(set_id) + ")")

        output_with_parallel_annotations = []

        for i in range(0, length+1):
            if i in index_to_strings:
                output_with_parallel_annotations.append(
                    "|".join(sorted(index_to_strings[i])))
            else:
                output_with_parallel_annotations.append("-")

        return output_with_parallel_annotations

    def write_antecedent_decisions_to_file(self, file):
        """ Write all antecedent decisions to a file.

        One decision is represented as one line in the file in the following
        format:
            folderid    part    anaphor_span    antecedent_span

        For example
            bn/voa/02/voa_0220  0   (10,11) (1,1)

        Args:
            file (file): The file to write antecedent decisions to.
        """
        for mention_span in sorted(self.antecedent_decisions.keys()):
            file.write(self.folder + self.id + "\t" +
                       self.part + "\t" +
                       str(mention_span) + "\t" +
                       str(self.antecedent_decisions[mention_span]) + "\n")

    def get_annotated_mentions_from_antecedent_decisions(self, span_pairs):
        """ Overwrite coreference attributes with information from span pairs.

        In particular, interpret the spans in span pairs as anaphor/antededent
        decisions and update attributes annotated_mentions,
        spans_to_annotated_mentions, coref accordingly.

        Args:
            span_pairs (list((Span,Span))): A list of span pairs corresponding
                to anaphor/antecedent decisions.
        """
        self.spans_to_annotated_mentions.clear()

        set_id = 0
        for span_anaphor, span_antecedent in span_pairs:
            if span_antecedent not in self.spans_to_annotated_mentions:
                self.spans_to_annotated_mentions[span_antecedent] = \
                    mentions.Mention.from_document(span_antecedent, self)
                self.spans_to_annotated_mentions[
                    span_antecedent].attributes["annotated_set_id"] = set_id
                set_id += 1
            if span_anaphor not in self.spans_to_annotated_mentions:
                self.spans_to_annotated_mentions[span_anaphor] = \
                    mentions.Mention.from_document(span_anaphor, self)
                self.spans_to_annotated_mentions[span_anaphor].attributes[
                    "annotated_set_id"] = self.spans_to_annotated_mentions[
                        span_antecedent].attributes["annotated_set_id"]

            self.spans_to_annotated_mentions[span_anaphor].attributes[
                "antecedent"] = self.spans_to_annotated_mentions[
                    span_antecedent]

            self.annotated_mentions = sorted(
                list(self.spans_to_annotated_mentions.values()))

            self.coref.clear()

            for span in self.spans_to_annotated_mentions:
                self.coref[span] = self.spans_to_annotated_mentions[
                    span].attributes["annotated_set_id"]
