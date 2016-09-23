""" Manage individual documents of text collections.

In particular, classes in this module allow to access the linguistic
information present in documents.
"""

from collections import defaultdict
import logging

from cort.core import mentions
from cort.core import spans

import StanfordDependencies

import nltk

logger = logging.getLogger(__name__)


__author__ = 'smartschat'


class Document(object):
    """Represents a document.

    Attributes:
        identifier (str): A unique identifier for the document.
        in_sentence_ids (list(int)): In-sentence indicies of all tokens in the
            document, for example [0, 1, 2, 0, 1, 2, 3, 4, ...]
        tokens (list(str)): All tokens.
        pos (list(str)): All part-of-speech tags.
        ner (list(str)): All named entity tags (if a token does not have a
            tag, the tag is set to NONE).
        parse (list(nltk:ParentedTree)): All parse trees.
        dep (list(list(StanfordDependencies.CoNLL.Token)): All dependencies
            represented as lists of tokens with label information and pointers
            to heads. One list for each sentence.
        speakers (list(str)): All speaker ids,
        coref (dict(span, int)): A mapping of mention spans to their
            coreference set id.
        annotated_mentions list(Mention): All annotated mentions.
        system_mentions list(Mention): The system mentions (initially empty).
    """
    def __init__(self, identifier, sentences, coref):
        """ Construct a document from sentence and coreference information.

        Args:
            identifier (str): A unique identifier for the document.
            sentences(list): A list of sentence information. The ith item
                contains information about the ith sentence. We assume that
                each ``sentences[i]`` is a 6-tuple
                ``tokens, pos, ner, speakers, parse, dep``, where

                * tokens (list(str)): All tokens in the sentence.
                * pos (list(str)): All part-of-speech tags in the sentence.
                * ner (list(str)): All named entity tags in the sentence (if a
                  token does not have a tag, the tag is set to NONE).
                * speakers (list(str)): All speaker ids in the sentence.
                * parse (str): A string representation of the sentence's parse
                  tree (should be readable by nltk)
                * dep (list(StanfordDependencies.CoNLL.Token): All dependencies
                  in the sentence represented as lists of tokens with label
                  information and pointers to heads.
            coref (dict(span, int)): A mapping of mention spans to their
            coreference set id.
        """
        self.identifier = identifier

        self.in_sentence_ids = []
        self.sentence_spans = []
        self.tokens = []
        self.pos = []
        self.ner = []
        self.parse = []
        self.dep = []
        self.speakers = []
        self.coref = coref

        for sentence in sentences:
            tokens, pos, ner, speakers, parse, dep = sentence

            offset = len(self.tokens)

            self.in_sentence_ids += list(range(0, len(tokens)))

            self.sentence_spans.append(spans.Span(
                offset, offset + len(tokens) - 1
            ))

            self.tokens += tokens
            self.pos += pos
            self.ner += ner
            self.parse.append(nltk.ParentedTree.fromstring(parse))
            self.dep.append(dep)
            self.speakers += speakers

        self.annotated_mentions = self.__get_annotated_mentions()
        self.system_mentions = []

    def __get_annotated_mentions(self):
        mention_spans = sorted(list(self.coref.keys()))

        seen = set()

        annotated_mentions = []

        for span in mention_spans:
            set_id = self.coref[span]
            annotated_mentions.append(
                mentions.Mention.from_document(
                    span, self, first_in_gold_entity=set_id not in seen
                )
            )
            seen.add(set_id)

        return annotated_mentions

    def __repr__(self):
        return self.identifier

    def __hash__(self):
        return hash(self.identifier)

    def __lt__(self, other):
        """ Check whether this document is less than another document.

        It is less than another document if and only if
        ``(self.folder, self.id, self.part) <
        (another.folder, another.id, another.part)``,
        where the folders and ids are compared lexicographically.

        Args:
            other (ConLLDocument): A document.

        Returns:
            True if this document is less than other, False otherwise.
        """
        return self.identifier < other.identifier

    def __eq__(self, other):
        """ Check for document equality.

        Two documents are considered equal is they have the same folder, id and
        the part.

        Args:
            other (ConLLDocument): A document.

        Returns:
            True if id and part of the documents are equal.
        """
        if isinstance(other, self.__class__):
            return self.identifier == other.identifier
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def write_antecedent_decisions_to_file(self, file):
        """ Write all antecedent decisions to a file.

        One decision is represented as one line in the file in the following
        format:

        ``identifier    anaphor_span    antecedent_span``

        For example ``(bc/cctv/00/cctv_0000); part 000   (10, 11) (1, 1)``

        Args:
            file (file): The file to write antecedent decisions to.
        """

        for mention in self.system_mentions:
            ante = mention.attributes["antecedent"]

            if ante:
                file.write(self.identifier + "\t" +
                           str(mention.span) + "\t" +
                           str(ante.span) + "\n")

    def get_annotated_mentions_from_antecedent_decisions(self, span_pairs):
        """ Overwrite coreference attributes with information from span pairs.

        In particular, interpret the spans in span pairs as anaphor/antededent
        decisions and update attributes annotated_mentions,
        spans_to_annotated_mentions, coref accordingly.

        Args:
            span_pairs (list((Span,Span))): A list of span pairs corresponding
                to anaphor/antecedent decisions.
        """

        spans_to_annotated_mentions = {}

        for mention in self.annotated_mentions:
            spans_to_annotated_mentions[mention.span] = mention

        set_id = 0
        for span_anaphor, span_antecedent in span_pairs:
            if span_antecedent not in spans_to_annotated_mentions:
                spans_to_annotated_mentions[span_antecedent] = \
                    mentions.Mention.from_document(span_antecedent, self)
                spans_to_annotated_mentions[
                    span_antecedent].attributes["annotated_set_id"] = set_id
                set_id += 1
            if span_anaphor not in spans_to_annotated_mentions:
                spans_to_annotated_mentions[span_anaphor] = \
                    mentions.Mention.from_document(span_anaphor, self)
                spans_to_annotated_mentions[span_anaphor].attributes[
                    "annotated_set_id"] = spans_to_annotated_mentions[
                        span_antecedent].attributes["annotated_set_id"]

            spans_to_annotated_mentions[span_anaphor].attributes[
                "antecedent"] = spans_to_annotated_mentions[
                    span_antecedent]

            self.annotated_mentions = sorted(
                list(spans_to_annotated_mentions.values()))

            self.coref.clear()

            for span in spans_to_annotated_mentions:
                self.coref[span] = spans_to_annotated_mentions[
                    span].attributes["annotated_set_id"]

    def get_antecedent_decisions(self, which_mentions="annotated"):
        """ Get all antecedent decisions in this document.

        Args:
            which_mentions (str): Either "annotated" or "system". Defaults to
                "system". Signals whether to consider annotated mentions or
                system mentions.

        Returns:
            dict(Mention, Mention): The mapping of antecedent decisions.
        """
        antecedent_decisions = {}

        doc_mentions = None

        if which_mentions == "annotated":
            doc_mentions = self.annotated_mentions
        elif which_mentions == "system":
            doc_mentions = self.system_mentions

        for mention in doc_mentions:
            antecedent = mention.attributes["antecedent"]

            if antecedent:
                antecedent_decisions[mention] = antecedent

        return antecedent_decisions

    def get_sentence_id_and_span(self, span):
        """ Get the sentence span from the sentence embedding the span.

        Args:
            span (Span): A span corresponding to a fragment of the document.

        Returns:
            Span: The span of the sentence which embeds the text corresponding
            to the span.
        """
        for i, sentence_span in enumerate(self.sentence_spans):
            if sentence_span.embeds(span):
                return i, sentence_span

    def to_simple_output(self):
        """ Convert the document into a simple textual representation,
        containing tokens and coreference information.

        In particular, the representation has one sentence per line, one space
        between each token. Each system mention is enclosed by
        ``<mention> ... </mention>`` tags, with the following attributes:

        * id (int): integer identifier of the mention. Equals position in the
            list of system mentions (excluding the dummy mention)
        * span_start (int): start of the mention span
        * span_end (int): end of the mention span
        * entity (int): set id of the mention. Only when the mention is in some
            coreference cluster
        * antecedent (int): id of the antecedent. Only if the mention has some
            antecedent.

        Returns:
            (str): A textual representation of document as described above.
        """
        content = []

        for sentence_span in self.sentence_spans:
            content += self.tokens[sentence_span.begin:
                                   sentence_span.end+1]

        mention_to_id = {}

        for i, mention in enumerate(self.system_mentions[1:]):
            mention_to_id[mention] = i

            tag_begin = "<mention id=\"" + str(i) + "\" " \
                        "span_start=\"" + str(mention.span.begin) + "\" " \
                        "span_end=\"" + str(mention.span.end) + "\""

            if mention.attributes["set_id"]:
                tag_begin += " entity=\"" + \
                             str(mention.attributes["set_id"]) + "\""

            if mention.attributes["antecedent"]:
                antecedent_id = mention_to_id[
                    mention.attributes["antecedent"]]

                tag_begin += " antecedent=\"" + str(antecedent_id) + "\""

            tag_begin += ">"

            old_begin = content[mention.span.begin]
            content[mention.span.begin] = tag_begin + old_begin

            content[mention.span.end] += "</mention>"

        output_string = ""

        for sentence_span in self.sentence_spans:
            output_string += " ".join(
                content[sentence_span.begin:sentence_span.end+1]) + "\n"

        return output_string

    def get_html_friendly_identifier(self):
        """ Transform the identifier to a HTML/CSS/JS-friendly representation
        for visualization.

        Returns:
            str: The HTML/CSS-JS-friendly representation.
        """
        return self.identifier.replace(".", "_").replace("/", "_")


class CoNLLDocument(Document):
    """Represents a document in CoNLL format.

    For a specification of the format, see
    http://conll.cemantix.org/2012/data.html.

    Attributes:
        identifier (str): A unique identifier for the document.
        in_sentence_ids (list(int)): In-sentence indicies of all tokens in the
            document, for example [0, 1, 2, 0, 1, 2, 3, 4, ...]
        tokens (list(str)): All tokens.
        pos (list(str)): All part-of-speech tags.
        ner (list(str)): All named entity tags (if a token does not have a
            tag, the tag is set to NONE).
        parse (list(nltk:ParentedTree)): All parse trees.
        dep (list(list(StanfordDependencies.CoNLL.Token)): All dependencies
            represented as lists of tokens with label information and pointers
            to heads. One list for each sentence.
        speakers (list(str)): All speaker ids,
        coref (dict(span, int)): A mapping of mention spans to their
            coreference set id.
        annotated_mentions list(Mention): All annotated mentions.
        system_mentions list(Mention): The system mentions (initially empty).
        document_table (list(list(str))): A tabular representation of the
            document (as in the CoNLL data).
    """

    def __init__(self, document_as_string):
        """ Construct a document from a string representation.

            The Format must follow the CoNLL format, see
                http://conll.cemantix.org/2012/data.html.

            Args:
                document_as_string (str): A representation of a document in
                    the CoNLL format.
            """
        identifier = " ".join(document_as_string.split("\n")[0].split(" ")[2:])

        self.document_table = CoNLLDocument.__string_to_table(
            document_as_string)
        in_sentence_ids = [int(i) for i in self.__extract_from_column(2)]
        indexing_start = in_sentence_ids[0]
        if indexing_start != 0:
            logger.warning("Detected " +
                           str(indexing_start) +
                           "-based indexing for tokens in sentences in input,"
                           "transformed to 0-based indexing.")
            in_sentence_ids = [i - indexing_start for i in in_sentence_ids]
        sentence_spans = CoNLLDocument.__extract_sentence_spans(in_sentence_ids)
        temp_tokens = self.__extract_from_column(3)
        temp_pos = self.__extract_from_column(4)
        temp_ner = self.__extract_ner()
        temp_speakers = self.__extract_from_column(9)
        coref = CoNLLDocument.__get_span_to_id(self.__extract_from_column(-1))
        parses = [CoNLLDocument.get_parse(span,
                                          self.__extract_from_column(5),
                                          temp_pos,
                                          temp_tokens)
                  for span in sentence_spans]
        sd = StanfordDependencies.get_instance()
        dep_trees = sd.convert_trees(
            [parse.replace("NOPARSE", "S") for parse in parses],
        )
        sentences = []
        for i, span in enumerate(sentence_spans):
            sentences.append(
                (temp_tokens[span.begin:span.end + 1],
                 temp_pos[span.begin:span.end + 1],
                 temp_ner[span.begin:span.end + 1],
                 temp_speakers[span.begin:span.end + 1],
                 parses[i],
                 dep_trees[i])
            )

        super(CoNLLDocument, self).__init__(identifier, sentences, coref)

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

    @staticmethod
    def __extract_sentence_spans(in_sentence_ids):
        sentence_spans = []

        span_start = 0

        for i in range(1, len(in_sentence_ids)):
            if in_sentence_ids[i] <= in_sentence_ids[i-1]:
                sentence_spans.append(spans.Span(span_start, i-1))
                span_start = i

        sentence_spans.append(spans.Span(span_start,
                                         len(in_sentence_ids)-1))

        return sentence_spans

    @staticmethod
    def __get_span_to_id(column):
        span_to_id = {}

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

    @staticmethod
    def get_parse(span, parses_as_string, pos, tokens):
        """ Get a the parse tree (as a string) of to the span.

        Args:
            span (Span): A span corresponding to a fragment of the document.

        Returns:
            str: A string representation of the parse tree of the span.
        """
        parse_tree = ""
        for i in range(span.begin, span.end+1):
            parse_bit = parses_as_string[i]
            parse_tree += \
                parse_bit.replace("(", " (").replace(
                    "*", " (" + pos[i] + " " + tokens[i] + ")")

        return parse_tree.strip()

    def get_string_representation(self):
        """ Get a string representation of the document.

        Returns:
            str: A string representation of the document which conforms to the
            CoNLL format specifications
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

        begin = ("#begin document " + self.identifier + "\n")

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

    def get_html_friendly_identifier(self):
        """ Transform the identifier to a HTML/CSS/JS-friendly representation
        for visualization.

        Returns:
            str: The HTML/CSS/JS-friendly representation.
        """
        splitted_by_whitespace = self.identifier.split()
        return splitted_by_whitespace[0].split("/")[-1][:-2] + \
               "_part_" + splitted_by_whitespace[-1]
