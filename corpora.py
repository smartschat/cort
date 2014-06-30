from documents import CoNLLDocument


__author__ = 'smartschat'


class Corpus:
    def __init__(self, description, documents):
        self.description = description
        self.documents = documents

    def __iter__(self):
        return iter(self.documents)

    @staticmethod
    def from_file(description, coref_file):
        if coref_file is None:
            return []

        documents = []

        current_document = ""

        for line in coref_file.readlines():
            if line.startswith("#begin") and current_document != "":
                doc = CoNLLDocument(current_document)
                documents.append(doc)
                current_document = ""
            current_document += line

        doc = CoNLLDocument(current_document)
        documents.append(doc)

        return Corpus(description, documents)

    def write_to_file(self, file):
        for document in self.documents:
            document.write_to_file(document.system_mentions, file)