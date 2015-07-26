""" Represent and manipulate text collections as a list of documents."""

from collections import defaultdict
import multiprocessing

from cort.analysis import data_structures
from cort.core import documents
from cort.core import spans

__author__ = 'smartschat'


def from_string(string):
    return documents.CoNLLDocument(string)


class Corpus:
    """Represents a text collection (a corpus) as a list of documents.

    Such a text collection can also be read from data, and be supplemented with
    antecedent information.

    Attributes:
        description(str): A human-readable description of the corpus.
        documents (list(Document)): A list of CoNLL documents.
    """

    def __init__(self, description, corpus_documents):
        """Construct a Corpus from a description and a list of documents.

        Args:
            description (str): A human-readable description of the corpus.
            documents (list(Document)): A list of documents.
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
            Corpus: A corpus consisting of the documents described in
            coref_file
        """

        if coref_file is None:
            return []

        document_as_strings = []

        current_document = ""

        for line in coref_file.readlines():
            if line.startswith("#begin") and current_document != "":
                document_as_strings.append(current_document)
                current_document = ""
            current_document += line

        document_as_strings.append(current_document)

        return Corpus(description, sorted([from_string(doc) for doc in
                                           document_as_strings]))



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

    def read_antecedents(self, file):
        """Augment corpus with antecedent decisions read from a file.

        The attribute annotated_mentions is overwritten by mentions read in
        from the antecedents file. Input files should have one antecedent
        decision per line, where entries are separated by tabs. The format is

            doc_identifier   (anaphor_start, anaphor_end)    (ante_start, ante_end)

        where
            - doc_id is the identifier in the first line of an CoNLL document
              after #begin document, such as (bc/cctv/00/cctv_0000); part 000
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
            span_anaphor = splitted[1]
            span_antecedent = splitted[2]
            doc_identifier_to_pairs[doc_id].append(
                (spans.Span.parse(span_anaphor), spans.Span.parse(
                    span_antecedent)))

        for doc in self.documents:
            pairs = sorted(doc_identifier_to_pairs[doc.identifier])
            doc.get_annotated_mentions_from_antecedent_decisions(pairs)

    def read_coref_decisions(self,
                             mention_entity_mapping,
                             antecedent_mapping=None):
        """Augment corpus with coreference and antecedent decisions..

        Set set_id attribute and antecedent information for system mentions.

        Args:
            mention_entity_mapping (dict(Mention, int)): A mapping of mentions
                to entity identifiers.
            antecedent_mapping (dict(Mention, Mention)): A mapping of mentions
                to their antecedent. Optional..
        """
        for doc in self.documents:
            for mention in doc.system_mentions:
                if mention in mention_entity_mapping:
                    mention.attributes["set_id"] = \
                        mention_entity_mapping[mention]
                    if antecedent_mapping and mention in antecedent_mapping:
                        antecedent = antecedent_mapping[mention]
                        mention.attributes['antecedent'] = antecedent
                        mention.document.antecedent_decisions[mention.span] = \
                            antecedent.span

    def get_antecedent_decisions(self, which_mentions="annotated"):
        """ Get all antecedent decisions in this corpus.

        Args:
            which_mentions (str): Either "annotated" or "system". Defaults to
                "system". Signals whether to consider annotated mentions or
                system mentions.

        Returns:
            StructuredCoreferenceAnalysis: A StructuredCoreferenceAnalysis
                containing all antecedent decisions. Can be accessed like a
                dict. If this is assigned a a variable ``x``, the
                decisions can be accessed via ``x[self.description][
                "decisions"]["all"]``, where ``self.description`` is the
                ``description`` attribute of the corpus (e.g. ``x["pair"][
                "decisions"]["all"])..
        """
        antecedent_decisions = {
            self.description: {
                "decisions": {
                    "all": {}
                }
            }
        }

        all_decisions = set()

        for doc in self.documents:
            doc_decisions = doc.get_antecedent_decisions(which_mentions)
            for ana, ante in doc_decisions.items():
                all_decisions.add((ana, ante))

        antecedent_decisions[self.description]["decisions"]["all"] = \
            data_structures.EnhancedSet(all_decisions)

        return data_structures.StructuredCoreferenceAnalysis(
            antecedent_decisions, {self.description: self}, None)

    def are_coreferent(self, m, n):
        """ Compute whether two mentions are coreferent in this corpus.

        One use case of this function is when ``m`` and ``n`` belong to a
        different corpus object, but you are interested in whether they are
        coreferent according to the annotation present in this corpus.

        Args:
            m (Mention): A mention.
            n (Mention): Another mention.

        Returns:
            True if ``m`` and ``n`` are coreferent according to the annotation
            present in this corpus, False otherwise.
        """
        if m.document != n.document:
            return False
        elif m.document not in self.documents:
            return False
        else:
            doc = self.documents[self.documents.index(m.document)]

            if m.span not in doc.spans_to_annotated_mentions or \
               n.span not in doc.spans_to_annotated_mentions:
                return False

            m_in_this_corpus = doc.spans_to_annotated_mentions[m.span]
            n_in_this_corpus = doc.spans_to_annotated_mentions[n.span]

            return m_in_this_corpus.is_coreferent_with(n_in_this_corpus)