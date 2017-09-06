""" Represent and manipulate text collections as a list of documents."""

from collections import defaultdict

from cort.core import documents
from cort.core import spans
from cort.core import mention_property_computer
from cort.core import head_finders

__author__ = 'smartschat'


class Corpus:
    """Represents a text collection (a corpus) as a list of documents.

    Such a text collection can also be read from data, and be supplemented with
    antecedent information.

    Attributes:
        description(str): A human-readable description of the corpus.
        documents (list(Document)): A list of CoNLL documents.
    """

    def __init__(self,
                 description,
                 corpus_documents):
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
    def from_file(description,
                  coref_file,
                  mention_property_computer=mention_property_computer.EnglishMentionPropertyComputer(head_finders.EnglishHeadFinder())):
        """Construct a new corpus from a description and a file.

        The file must contain documents in the format for the CoNLL shared
        tasks on coreference resolution, see
        http://conll.cemantix.org/2012/data.html.

        Args:
            description (str): A human-readable description of the corpus.
            coref_file (file): A text file of documents in the CoNLL format.
            mention_property_computer (MentionPropertyComputer): An `MentionPropertyComputer`
                object that computes properties (such as number or gender) for mentions.
                Defaults to EnglishMentionPropertyComputer initialized with Collins' head finder.

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

        corpus = Corpus(description,
                        sorted([documents.CoNLLDocument(doc)
                                for doc in document_as_strings]))

        for doc in corpus:
            if mention_property_computer.needs_parse_trees():
                doc.enrich_with_parse_trees()

            if mention_property_computer.needs_dependency_trees():
                doc.enrich_with_dependency_trees()

            for i, mention in enumerate(doc.annotated_mentions):
                mention_property_computer.compute_properties(mention, i)

        return corpus

    def write_to_file(self, file, include_singletons=False):
        """Write a string representation of the corpus to a file,

        Args:
            file (file): The file the corpus should be written to
            include_singletons (bool): If True, include mentions which are not
                assigned any set id in the output. Defaults to False.            .
        """
        for document in self.documents:
            file.write(document.get_string_representation(include_singletons))

    def write_antecedent_decisions_to_file(self, file):
        """Write antecedent decisions in the corpus to a file.

        For the format, have a look at the documentation for
        read_antecedent_decisions in this class.

        Args:
            file (file): The file the antecedent decisions should be written
                to.
        """
        for document in self.documents:
            document.write_antecedent_decisions_to_file(file)

    def read_antecedents(self, file, mention_property_computer=mention_property_computer.EnglishMentionPropertyComputer(head_finders.EnglishHeadFinder())):
        """Augment corpus with antecedent decisions read from a file.

        The attribute `annotated_mentions` is overwritten by mentions read in
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
            file (file): The file the antecedent decisions should be read from.
            mention_property_computer (MentionPropertyComputer): An `MentionPropertyComputer`
                object that computes properties (such as number or gender) for mentions.
                Defaults to EnglishMentionPropertyComputer initialized with Collins' head finder.
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
            for i, mention in enumerate(doc.annotated_mentions, 1):
                mention_property_computer.compute_properties(mention, i)

    def read_coref_decisions(self,
                             union,
                             antecedent_mapping=None):
        """Augment corpus with coreference and antecedent decisions..

        Set set_id attribute and antecedent information for system mentions.

        Args:
            mention_entity_mapping (dict(Mention, int)): A mapping of mentions
                to entity identifiers.
            antecedent_mapping (dict(Mention, Mention)): A mapping of mentions
                to their antecedent. Optional..
        """
        mention_entity_mapping = {}

        representant_to_cluster = union.get_repr_to_cluster()

        for i, cluster in enumerate(sorted(representant_to_cluster.values())):
            for mention in cluster:
                mention_entity_mapping[mention] = i

        for doc in self.documents:
            for mention in doc.system_mentions:
                if mention in mention_entity_mapping:
                    mention.attributes["set_id"] = \
                        mention_entity_mapping[mention]
                    if antecedent_mapping and mention in antecedent_mapping:
                        mention.attributes['antecedent'] = antecedent_mapping[mention]
                        mention.document.antecedent_decisions[mention.span] = \
                            [antecedent.span for antecedent in antecedent_mapping[mention]]

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
