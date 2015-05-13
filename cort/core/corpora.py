""" Represent and manipulate text collections as a list of documents."""

from collections import defaultdict

from cort.core import documents
from cort.core import spans

__author__ = 'smartschat'


class Corpus:
    """Represents a text collection (a corpus) as a list of documents.

    Such a text collection can also be read from data, and be supplemented with
    antecedent information.

    Attributes:
        description(str): A human-readable description of the corpus.
        documents (list(CoNLLDocument)): A list of CoNLL documents.
    """

    def __init__(self, description, corpus_documents):
        """Construct a Corpus from a description and a list of documents.

        Args:
            description (str): A human-readable description of the corpus.
            documents (list(CoNLLDocument)): A list of CoNLL documents.
        """
        self.description = description
        self.documents = corpus_documents

    def __iter__(self):
        """Return an iterator over documents in the corpus.

        Returns:
            An iterator over CoNLLDocuments.
        """
        return iter(self.documents)

    @staticmethod
    def from_file(description, coref_file):
        """Construct a new corpus from a description and a file.

        The file must contain documents in the format for the CoNLL shared
        tasks on coreference resolution, see
        http://conll.cemantix.org/2012/data.html.

        Args:
            description (str): A human-readable description of the corpus.
            coref_file (file): A text file of documents in the CoNLL format.

        Returns:
            (Corpus): A corpus consisting of the documents described in
                coref_file
        """

        if coref_file is None:
            return []

        documents_from_file = []

        current_document = ""

        for line in coref_file.readlines():
            if line.startswith("#begin") and current_document != "":
                doc = documents.CoNLLDocument(current_document)
                documents_from_file.append(doc)
                current_document = ""
            current_document += line

        doc = documents.CoNLLDocument(current_document)
        documents_from_file.append(doc)

        return Corpus(description, sorted(documents_from_file))

    def write_to_file(self, file):
        """Write a string representation of the corpus to a file,

        Args:
            file (file): The file the corpus should be written to.
        """
        for document in self.documents:
            file.write(document.get_string_representation())

    def write_antecedent_decisions_to_file(self, file):
        """Write antecedent decisions in the corpus to a file.

        For the format, have a look at the documenation for
        read_antecedent_decisions in this class.

        Args:
            file (file): The file the antecedent decisions should be written
                to.
        """
        for document in self.documents:
            document.write_antecedent_decisions_to_file(file)

    def get_genre_to_doc_map(self):
        """Return a map from genre identifiers to a documents of such genre.

        Returns:
            defaultdict(str, list)): A mapping from genres to
                a list of documents.
        """
        genre_to_doc = defaultdict(list)
        for doc in self.documents:
            genre_to_doc[doc.genre].append(doc)
        return genre_to_doc

    def read_antecedents(self, file):
        """Augment corpus with antecedent decisions read from a file.

        The attribute annotated_mentions is overwritten by mentions read in
        from the antecedents file. Input files should have one antecedent
        decision per line, where entries are separated by tabs. The format is

            doc_id  doc_part    (anaphor_start, anaphor_end)    (ante_start, ante_end)

        where
            - doc_id is the id as in the first column of the CoNLL original
              data,
            - doc_part is the part number (with trailing 0s),
            - anaphor_start is the position in the document where the anaphor
              begins (counting from 0),
            - anaphor_end is the position where the anaphor ends (inclusive),
            - ante_start, ante_end analogously for the antecedent.

        Args:
            file (file): The file the antecedent decisions should be written
                to.
        """
        doc_identifier_to_pairs = defaultdict(list)
        for line in file.readlines():
            splitted = line.split("\t")
            doc_id = splitted[0]
            part_id = splitted[1]
            span_anaphor = splitted[2]
            span_antecedent = splitted[3]
            doc_identifier_to_pairs[(doc_id, part_id)].append(
                (spans.Span.parse(span_anaphor), spans.Span.parse(
                    span_antecedent)))

        for doc in self.documents:
            pairs = sorted(doc_identifier_to_pairs[
                (doc.folder + doc.id, doc.part)])
            doc.get_annotated_mentions_from_antecedent_decisions(pairs)
