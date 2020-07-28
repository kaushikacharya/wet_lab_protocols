#!/usr/bin/env python

"""
References:
----------
https://spacy.io/api/sentencizer
    Examples show how to use Sentencizer.

https://spacy.io/usage/linguistic-features#sbd
    Default segmentation approach uses dependency parsing.
    There's also option to create custom/rule based segmentation.

    Referred by Ines in https://github.com/explosion/spaCy/issues/1756#issuecomment-387225769
"""

import argparse
import os
import spacy
from spacy import displacy
from spacy.lang.en import English
from spacy.pipeline import Sentencizer


class NLPProcess:
    """NLP processing of text
    """
    def __init__(self, model="en_core_web_sm"):
        self.model = model
        self.nlp = None
        self.nlp_sentencizer = None

    def load_nlp_model(self, verbose=False):
        self.nlp = spacy.load(name=self.model)
        if verbose:
            print("Model: {} loaded".format(self.model))
            print("pipe names: {}".format(self.nlp.pipe_names))

    def build_sentencizer(self, verbose=False):
        self.nlp_sentencizer = English()
        sentencizer = Sentencizer()
        self.nlp_sentencizer.add_pipe(component=sentencizer)
        if verbose:
            print("pipe names: {}".format(self.nlp_sentencizer.pipe_names))

    def sent_tokenize(self, text):
        """Segment text into sentences."""
        assert self.nlp_sentencizer is not None, "pre-requisite: Execute build_sentencizer()"
        # TODO spaCy sentencizer fails in few cases e.g. protocol_101 (2nd line). Would need additional processing.
        doc = self.nlp_sentencizer(text)
        return doc.sents

    def construct_doc(self, text):
        """Construct Doc container from the text.

            Reference:
            ---------
            https://spacy.io/api/doc
        """
        assert self.nlp is not None, "pre-requisite: Execute load_nlp_model()"
        doc = self.nlp(text)
        return doc

def main(args, verbose=True):
    file_document = os.path.join(args.data_dir, "Standoff_Format/protocol_" + args.protocol_id + ".txt")

    obj_nlp_process = NLPProcess()
    obj_nlp_process.load_nlp_model(verbose=True)
    obj_nlp_process.build_sentencizer(verbose=True)

    output_dir = os.path.join(os.path.dirname(__file__), "../output/debug", args.protocol_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # iterate over the protocol steps i.e. lines of the protocol file
    with open(file_document, encoding="utf-8") as fd_doc:
        sent_i = 0
        for line in fd_doc:
            line = line.strip()
            if line == "":
                continue

            sents = obj_nlp_process.sent_tokenize(text=line)

            for sent in sents:
                # sent type: Span (https://spacy.io/api/span)
                sent_text = sent.text

                doc_sentence = obj_nlp_process.construct_doc(text=sent_text)

                if verbose:
                    svg = displacy.render(doc_sentence, style="dep")

                    svg_sentence_file = os.path.join(output_dir, str(sent_i) + ".svg")
                    with open(svg_sentence_file, mode="w", encoding="utf-8") as fd:
                        fd.write(svg)

                # increment sentence index
                sent_i += 1

if __name__ == "__main__":
    """
    nlp = English()
    print(nlp.pipe_names)
    sentencizer = Sentencizer()
    nlp.add_pipe(sentencizer)
    print(nlp.pipe_names)
    doc = nlp("Login to the hpc (at the University of Arizona) with your user name Create a directory for the pipeline Go to the hpc-blast directory and create sub directories to hold std-err, std-out, data, blastdb, blast-results and scripts.")
    print(list(doc.sents))
    for token in doc:
        print(token.text, token.pos_)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol_id", action="store", dest="protocol_id")
    parser.add_argument("--data_dir", action="store", default="C:/KA/lib/WNUT_2020/data/train_data/", dest="data_dir")

    args = parser.parse_args()

    main(args=args)

