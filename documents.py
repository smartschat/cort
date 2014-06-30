from collections import defaultdict
import re
from nltk import ParentedTree
from mentions import Mention
from spans import Span
import mention_extractor


__author__ = 'smartschat'


class CoNLLDocument:
    def __init__(self, document_as_string):
        begin = document_as_string.split("\n")[0]
        self.folder = CoNLLDocument.extract_folder(begin)
        self.id = CoNLLDocument.extract_id(begin)
        self.part = CoNLLDocument.extract_part(begin)
        self.genre = self.get_genre()

        self.document_table = CoNLLDocument.string_to_table(document_as_string)

        self.in_sentence_ids = [int(i) for i in self.extract_from_column(2)]
        self.tokens = self.extract_from_column(3)
        self.pos = self.extract_from_column(4)
        self.ner = self.extract_ner()
        self.parse = self.extract_from_column(5)
        self.speakers = self.extract_from_column(9)
        self.sentence_spans_to_id = self.extract_sentence_spans()

        self.coref = CoNLLDocument.get_span_to_id(self.extract_from_column(-1))
        for span in self.coref.keys():
            self.coref[span] = self.coref[span][0]

        # maps spans to mention objects
        self.spans_to_annotated_mentions = self.get_span_to_annotated_mentions()
        self.annotated_mentions = sorted(list(self.spans_to_annotated_mentions.values()))

        self.system_mentions = []

    def __repr__(self):
        return self.id + ", part " + str(self.part)

    def __hash__(self):
        return hash((self.id, self.part))

    @staticmethod
    def extract_id(document_begin):
        return document_begin.split()[2].split("/")[-1][0:-2]

    @staticmethod
    def extract_part(document_begin):
        return int(document_begin.split()[-1])

    @staticmethod
    def extract_folder(document_begin):
        return "/".join(document_begin.split()[2].split("/")[0:-1])[1:] + "/"

    def extract_from_column(self, column):
        entries = []
        for line in self.document_table:
            entries.append(line[column])

        return entries

    def extract_ner(self):
        entries = self.extract_from_column(10)
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
    def string_to_table(document_as_string):
        table = []

        document_contents = document_as_string.split("\n")[1:-2]

        for line in document_contents:
            if line != "" and not line.isspace():
                table.append(line.split())

        return table

    def extract_sentence_spans(self):
        sentence_spans_to_id = {}
        sentence_id = 0

        span_start = 0

        for i in range(1, len(self.in_sentence_ids)):
            if self.in_sentence_ids[i] <= self.in_sentence_ids[i-1]:
                sentence_spans_to_id[Span(span_start, i-1)] = sentence_id
                sentence_id += 1
                span_start = i

        sentence_spans_to_id[Span(span_start, len(self.in_sentence_ids)-1)] = sentence_id

        return sentence_spans_to_id

    def get_parse(self, span):
        parse_tree = ""
        for i in range(span.begin, span.end+1):
            parse_bit = self.parse[i]
            parse_tree += parse_bit.replace("(", " (").replace("*", " (" + self.pos[i] + " " + self.tokens[i] + ")")

        return parse_tree.strip()

    def get_embedding_sentence(self, span):
        for sentence_span in self.sentence_spans_to_id.keys():
            if sentence_span.embeds(span):
                return sentence_span

    @staticmethod
    def get_span_to_id(column):
        span_to_id = defaultdict(list)

        ids_to_stack = defaultdict(list)

        for i in range(0, len(column)):
            entry = column[i]

            if entry != "-":
                parallel_annotations = entry.split("|")

                for annotation in parallel_annotations:
                    if annotation.startswith("(") and annotation.endswith(")"):
                        set_id = annotation[1:-1]
                        span_to_id[Span(i,i)].append(int(set_id))
                    elif annotation.startswith("("):
                        set_id = annotation[1:]
                        ids_to_stack[set_id].append(i)
                    elif annotation.endswith(")"):
                        set_id = annotation[:-1]
                        span_to_id[Span(ids_to_stack[set_id].pop(), i)].append(int(set_id))

        return span_to_id

    def get_span_to_annotated_mentions(self):
        spans = self.coref.keys()

        spans = sorted(spans)
        span_to_mentions = {}

        for span in spans:
            span_to_mentions[span] = Mention.from_document(span, self)

        return span_to_mentions

    @staticmethod
    def parse_span(span, column):
        entry_begin = column[span.begin]
        entry_end = column[span.end]

        if entry_begin != "-" and entry_end != "-":
            parallel_annotations_begin = entry_begin.split("|")
            parallel_annotations_end = entry_end.split("|")

            for annotation_begin in parallel_annotations_begin:
                for annotation_end in parallel_annotations_end:
                    if annotation_begin.startswith("(") and annotation_end.endswith(")"):
                        id_begin = int(annotation_begin.replace("(", "").replace(")", ""))
                        id_end = int(annotation_end.replace("(", "").replace(")", ""))
                        if id_begin == id_end:
                            return id_begin

    def are_coreferent(self, m, n):
        return m.document == n.document \
            and m.document == self \
            and m.span in self.spans_to_annotated_mentions \
            and n.span in self.spans_to_annotated_mentions \
            and self.spans_to_annotated_mentions[m.span].attributes["annotated_set_id"] == self.spans_to_annotated_mentions[n.span].attributes["annotated_set_id"]

    def get_genre(self):
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

    def __iter__(self):
        return iter(self.annotated_mentions)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id and self.part == other.part
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def extract_system_mention_spans(self):
        mention_spans = []
        for sentence_span in self.sentence_spans_to_id:
            sentence_tree = ParentedTree(self.get_parse(sentence_span))
            in_sentence_spans = mention_extractor.extract_mention_spans(sentence_tree, self.ner[sentence_span.begin:sentence_span.end+1])
            mention_spans += [Span(sentence_span.begin + span.begin, sentence_span.begin + span.end) for span in in_sentence_spans]

        return sorted(mention_spans)

    def extract_system_mentions(self):
        mentions = mention_extractor.post_process_mentions([Mention.from_document(span, self) for span in self.extract_system_mention_spans()])

        # update set id
        for mention in mentions:
            mention.attributes["set_id"] = None

        self.system_mentions = mentions

    @staticmethod
    def get_string_representation_of_mentions(length, mentions):
        index_to_strings = defaultdict(list)

        for mention in mentions:
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
                output_with_parallel_annotations.append("|".join(sorted(index_to_strings[i])))
            else:
                output_with_parallel_annotations.append("-")

        return output_with_parallel_annotations

    def get_new_table_repr(self, mentions):
        mention_string_representation = CoNLLDocument.get_string_representation_of_mentions(len(self.document_table), mentions)

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

        return "\n".join(["\t".join(row) for row in padded_table])

    def write_to_file(self, mentions, file):
        updated_table_repr = self.get_new_table_repr(mentions)

        file.write("#begin document (" + self.folder + self.id + "); part " + str(self.part).zfill(3) + "\n")
        file.write(updated_table_repr)
        file.write("\n")
        file.write("#end document\n")