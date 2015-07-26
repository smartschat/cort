__author__ = 'martscsn'

import cort

import codecs

import stanford_corenlp_pywrapper

from StanfordDependencies import CoNLL

from cort.core import corpora, documents, spans

import bs4


class Pipeline():
    def __init__(self, corenlp_location, with_coref=False):
        package_dir = cort.__path__[0]

        if with_coref:
            self.proc = stanford_corenlp_pywrapper.CoreNLP(
                configfile=package_dir + "/config_files/corenlp_with_coref.ini",
                corenlp_jars=[corenlp_location + "/*"]
            )
        else:
            self.proc = stanford_corenlp_pywrapper.CoreNLP(
                configfile=package_dir + "/config_files/corenlp.ini",
                corenlp_jars=[corenlp_location + "/*"]
            )

        self.with_coref = with_coref

    def run_on_docs(self, identifier, docs):
        processed_documents = []

        for doc in docs:
            processed_documents.append(self.run_on_doc(
                codecs.open(doc, "r", "utf-8")
            ))

        return corpora.Corpus(identifier, processed_documents)

    def run_on_doc(self, doc_file, name=None):
        if self.with_coref:
            soup = bs4.BeautifulSoup(doc_file.read())
            preprocessed = self.proc.parse_doc(soup.text)
        else:
            data = doc_file.read()
            preprocessed = self.proc.parse_doc(data)

        sentences = []

        for sentence in preprocessed["sentences"]:
            processed_ner = []
            for ner in sentence["ner"]:
                if ner == "O" or ner == "MISC":
                    processed_ner.append("NONE")
                else:
                    processed_ner.append(ner)

            processed_dep = []

            index_to_dep_info = {}
            for dep_info in sentence["deps_basic"]:
                label, head, in_sent_index = dep_info
                index_to_dep_info[in_sent_index] = label, head

            for i in range(0, len(sentence["tokens"])):
                if i in index_to_dep_info.keys():
                    label, head = index_to_dep_info[i]
                    processed_dep.append(
                        CoNLL.Token(
                            form=sentence["tokens"][i],
                            lemma=sentence["lemmas"][i],
                            pos=sentence["pos"][i],
                            index=i+1,
                            head=head+1,
                            deprel=label,
                            cpos=None,
                            feats=None,
                            phead=None,
                            pdeprel=None
                        )
                    )
                else:
                    processed_dep.append(
                        CoNLL.Token(
                            form=sentence["tokens"][i],
                            lemma=sentence["lemmas"][i],
                            pos=sentence["pos"][i],
                            index=i+1,
                            head=0,
                            deprel="punc",
                            cpos=None,
                            feats=None,
                            phead=None,
                            pdeprel=None
                        )
                    )

            sentences.append(
                (sentence["tokens"],
                 sentence["pos"],
                 processed_ner,
                 ["-"]*len(sentence["tokens"]),
                 sentence["parse"],
                 processed_dep,
                )
            )

        if not name:
            name = doc_file.name

        if self.with_coref:
            antecedent_decisions = {}
            coref = {}

            mention_id_to_spans = {}

            max_entity = 0

            for mention in soup.findAll("mention"):
                if mention.get("entity"):
                    max_entity = max(max_entity, int(mention.get("entity")))

            for mention in soup.findAll("mention"):
                mention_id = int(mention.get("id"))

                span = spans.Span(int(mention.get("span_start")),
                                  int(mention.get("span_end")))

                mention_id_to_spans[mention_id] = span

                if mention.get("entity"):
                    annotated_set_id = int(mention.get("entity"))
                else:
                    annotated_set_id = max_entity + 1 + mention_id

                coref[span] = annotated_set_id

                if mention.get("antecedent"):
                    antecedent_decisions[span] = mention_id_to_spans[
                        int(mention.get("antecedent"))
                    ]

            doc = documents.Document(
                name,
                sentences,
                coref)

            spans_to_annotated_mentions = {}

            for mention in doc.annotated_mentions:
                spans_to_annotated_mentions[mention.span] = mention

            for span in antecedent_decisions:
                ante_span = antecedent_decisions[span]
                ana = spans_to_annotated_mentions[span]
                ante = spans_to_annotated_mentions[ante_span]
                ana.attributes["antecedent"] = ante
        else:
            doc = documents.Document(
                name,
                sentences,
                {})

        return doc
