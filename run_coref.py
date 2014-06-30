from __future__ import print_function
import sys
import logging
from corpora import Corpus
from decoders import MultigraphDecoder
import features

if len(sys.argv) != 3:
    print("usage:\n" + \
          "\t$ python run_coref.py input.conll output.conll\n" + \

          "input.conll should be following the format of the CoNLL shared tasks on coreference resolution (see\n" + \
          "http://conll.cemantix.org/2012/data.html).)")
    sys.exit(0)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.info("Reading in corpus")

corpus = Corpus.from_file("my corpus", open(sys.argv[1]))

logging.info("Extracting system mentions")

for doc in corpus:
    doc.extract_system_mentions()

negative_features = [features.not_modifier,
                     features.not_compatible,
                     features.not_embedding,
                     features.not_speaker,
                     features.not_pronoun_distance,
                     features.not_anaphoric]

positive_features = [features.alias,
                     features.non_pronominal_string_match,
                     features.head_match,
                     features.pronoun_same_canonical_form,
                     features.anaphor_pronoun,
                     features.speaker,
                     features.pronoun_parallelism]

logging.info("Decoding")

decoder = MultigraphDecoder(positive_features, negative_features)

decoder.decode(corpus)

corpus.write_to_file(open(sys.argv[2], "w"))

logging.info("Finished")
