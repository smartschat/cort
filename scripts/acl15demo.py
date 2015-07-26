#!/usr/bin/env python

from __future__ import print_function
import io
import logging
import pickle
import numpy

import pyximport
pyximport.install(setup_args={"include_dirs": numpy.get_include()})

from cort.preprocessing import pipeline
from cort.core import mention_extractor
from cort.coreference.approaches import mention_ranking
from cort.coreference import cost_functions, clusterer
from cort.coreference import experiments
from cort.coreference import features
from cort.coreference import instance_extractors
from cort.core import corpora
from cort.analysis import visualization, error_extractors, spanning_tree_algorithms

try:
    import tkinter as tki
except ImportError:
    import Tkinter as tki

__author__ = 'smartschat'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(''message)s')

class LiveDemo():
    def __init__(self):
        mention_features = [
            features.fine_type,
            features.gender,
            features.number,
            features.sem_class,
            features.deprel,
            features.head_ner,
            features.length,
            features.head,
            features.first,
            features.last,
            features.preceding_token,
            features.next_token,
            features.governor,
            features.ancestry
        ]

        pairwise_features = [
            features.exact_match,
            features.head_match,
            features.same_speaker,
            features.alias,
            features.sentence_distance,
            features.embedding,
            features.modifier,
            features.tokens_contained,
            features.head_contained,
            features.token_distance
        ]

        self.extractor = instance_extractors.InstanceExtractor(
            mention_ranking.extract_substructures,
            mention_features,
            pairwise_features,
            cost_functions.null_cost
        )

        logging.info("Loading model.")

        priors, weights = pickle.load(open("latent-model-train.obj", "rb"))

        self.perceptron = mention_ranking.RankingPerceptron(
            priors=priors,
            weights=weights,
            cost_scaling=0
        )

        logging.info("Loading CoreNLP models.")
        self.p = pipeline.Pipeline(
            "/home/sebastian/Downloads/stanford-corenlp-full-2015-04-20")

        self.root = tki.Tk()
        self.root.title("cort Demo")

        # create a Frame for the Text and Scrollbar
        self.txt_frm = tki.Frame(self.root, width=400, height=200)
        self.txt_frm.pack(fill="both", expand=True)

        # ensure a consistent GUI size
        self.txt_frm.grid_propagate(False)

        # implement stretchability
        self.txt_frm.grid_rowconfigure(0, weight=1)
        self.txt_frm.grid_columnconfigure(0, weight=1)

        # create a Text widget
        self.txt = tki.Text(self.txt_frm, borderwidth=3, relief="sunken")
        self.txt.config(font=("consolas", 12), undo=True, wrap='word')
        self.txt.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # create a Scrollbar and associate it with txt
        scrollb = tki.Scrollbar(self.txt_frm, command=self.txt.yview)
        scrollb.grid(row=0, column=1, sticky='nsew')
        self.txt['yscrollcommand'] = scrollb.set

        self.button = tki.Button(self.root, text='Resolve Coreference',
                            command=self.do_coreference)

        self.button.pack()

    def run(self):
        self.root.mainloop()

    def do_coreference(self):
        testing_corpus = corpora.Corpus("input", [self.p.run_on_doc(
            io.StringIO(self.txt.get("0.0", tki.END)), "input")])

        logging.info("Extracting system mentions.")
        for doc in testing_corpus:
            doc.system_mentions = mention_extractor.extract_system_mentions(doc)

        mention_entity_mapping, antecedent_mapping = experiments.predict(
            testing_corpus,
            self.extractor,
            self.perceptron,
            clusterer.all_ante
        )

        testing_corpus.read_coref_decisions(mention_entity_mapping, antecedent_mapping)

        logging.info("Visualize")

        for doc in testing_corpus:
            max_id = 0

            for mention in doc.system_mentions[1:]:
                set_id = mention.attributes["set_id"]

                if set_id:
                    max_id = max(set_id, max_id)

            max_id += 1

            doc.annotated_mentions = []

            for i, mention in enumerate(doc.system_mentions[1:]):
                if mention.attributes["set_id"]:
                    mention.attributes["annotated_set_id"] = mention.attributes[
                        "set_id"]
                else:
                    mention.attributes["annotated_set_id"] = max_id + i
                doc.annotated_mentions.append(mention)

        ex = error_extractors.ErrorExtractor(testing_corpus,
                                         spanning_tree_algorithms.recall_accessibility,
                                         spanning_tree_algorithms.precision_system_output)

        ex.add_system(testing_corpus)

        decisions = ex.get_errors()

        visualizer = visualization.Visualizer(decisions, "input",
                                              for_raw_input=True)

        visualizer.run()

demo = LiveDemo()

demo.run()
