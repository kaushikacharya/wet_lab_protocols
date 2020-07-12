#!/usr/bin/env python

import os
import spacy
from spacy import displacy

from .dataset import *


class Feature:
    def __init__(self):
        self.nlp = None

    def load_nlp_model(self, model="en_core_web_sm", verbose=False):
        self.nlp = spacy.load(name=model)
        if verbose:
            print("Model: {} loaded".format(model))

    def extract_document_features(self, doc_obj, verbose=False):
        """Extract features for the document.

            Parameters:
            ----------
            doc_obj : Document class object(dataset.py)
        """
        assert self.nlp is not None, "pre-requisite: spacy model should be loaded"

        output_dir = os.path.join(os.path.dirname(__file__), "../output/debug", str(doc_obj.id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # iterate over each of the sentences
        for sent_i in range(len(doc_obj.sentences)):
            start_char_pos = doc_obj.sentences[sent_i].start_char_pos
            end_char_pos = doc_obj.sentences[sent_i].end_char_pos
            sentence_text = doc_obj.text[start_char_pos: end_char_pos]

            doc_sentence = self.nlp(sentence_text)

            if verbose:
                svg = displacy.render(doc_sentence, style="dep")

                svg_sentence_file = os.path.join(output_dir, str(sent_i)+".svg")
                with open(svg_sentence_file, mode="w", encoding="utf-8") as fd:
                    fd.write(svg)

if __name__ == "__main__":
    feature_obj = Feature()
    feature_obj.load_nlp_model(verbose=True)

    protocol_id = 3
    data_folder = "C:/KA/lib/WNUT_2020/data/train_data/"
    file_document = os.path.join(data_folder, "Standoff_Format/protocol_" + str(protocol_id) + ".txt")
    file_conll_ann = os.path.join(data_folder, "Conll_Format/protocol_" + str(protocol_id) + "_conll.txt")

    document_obj = Document(doc_id=protocol_id)
    document_obj.parse_document(document_file=file_document)
    document_obj.parse_conll_annotation(conll_ann_file=file_conll_ann)

    feature_obj.extract_document_features(doc_obj=document_obj, verbose=True)

