#!/usr/bin/env python

"""
Feature extraction for a Document

Reference:
---------
https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html
"""

import argparse
import spacy
from spacy import displacy

from .dataset import *


class Feature:
    """Extraction of features of a Document class(dataset.py).
    """
    def __init__(self, doc_obj):
        """
            Parameters:
            ----------
            doc_obj : Document class object(dataset.py)
        """
        self.doc_obj = doc_obj

    def extract_document_features(self, verbose=False):
        """Extract features for the document.
        """
        # document features: List of sentence features
        # sentence feature: List of word features stored in dict
        doc_features = []

        # iterate over each of the sentence
        for sent_index in range(len(self.doc_obj.sentences)):
            if verbose:
                start_char_pos = self.doc_obj.sentences[sent_index].start_char_pos
                end_char_pos = self.doc_obj.sentences[sent_index].end_char_pos
                sent_text = self.doc_obj.text[start_char_pos: end_char_pos]
                start_word_index = self.doc_obj.sentences[sent_index].start_word_index
                end_word_index = self.doc_obj.sentences[sent_index].end_word_index
                start_token_index = self.doc_obj.sentences[sent_index].start_token_index
                end_token_index = self.doc_obj.sentences[sent_index].end_token_index

                print("\nSentence #{}: char pos range: ({}, {}) :: token index range: ({}, {}) :: "
                      "word index range: ({}, {}) :: text: {}".format(
                    sent_index, start_char_pos, end_char_pos, start_token_index, end_token_index, start_word_index,
                    end_word_index, sent_text))

            sent_features = self.extract_sentence_features(sent_index=sent_index)
            doc_features.append(sent_features)

        return doc_features

    def extract_sentence_features(self, sent_index):
        """Extract sentence features.

            Parameters:
            ----------
            sent_index : int

            Returns
            -------
            List of word features
        """
        sent_features = []

        start_word_index = self.doc_obj.sentences[sent_index].start_word_index
        end_word_index = self.doc_obj.sentences[sent_index].end_word_index

        for word_index in range(start_word_index, end_word_index):
            word_obj = self.doc_obj.words[word_index]

            word_features = dict()
            word_features["bias"] = 1.0
            word_text = self.doc_obj.text[word_obj.start_char_pos: word_obj.end_char_pos]
            word_features["word_lower"] = word_text.lower()

            if word_index == start_word_index:
                word_features["BOS"] = True
            else:
                word_features["prev_word_lower"] = sent_features[-1]["word_lower"]

            if word_obj.end_token_index - word_obj.start_token_index > 1:
                word_features["multi_token"] = True

            # TODO In case of multi-tokens, select syntactic features from the prime token
            word_features["POS"] = self.doc_obj.tokens[word_obj.start_token_index].part_of_speech
            word_features["dependency"] = self.doc_obj.tokens[word_obj.start_token_index].dependency_tag

            # append word features to the list of sentence features
            sent_features.append(word_features)

        return sent_features

def main(args):
    import os
    from .nlp_process import NLPProcess

    file_document = os.path.join(args.data_dir, "Standoff_Format/protocol_" + args.protocol_id + ".txt")
    file_conll_ann = os.path.join(args.data_dir, "Conll_Format/protocol_" + args.protocol_id + "_conll.txt")

    obj_nlp_process = NLPProcess()
    obj_nlp_process.load_nlp_model(verbose=True)
    obj_nlp_process.build_sentencizer(verbose=True)

    document_obj = Document(doc_id=int(args.protocol_id), nlp_process_obj=obj_nlp_process)
    document_obj.parse_document(document_file=file_document)
    document_obj.parse_conll_annotation(conll_ann_file=file_conll_ann)

    feature_obj = Feature(doc_obj=document_obj)
    doc_features = feature_obj.extract_document_features(verbose=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol_id", action="store", dest="protocol_id")
    parser.add_argument("--data_dir", action="store", default="C:/KA/lib/WNUT_2020/data/train_data/", dest="data_dir")

    args = parser.parse_args()
    main(args=args)

"""
Tasks:
    - Features hierarchy:
        - Document -> Line/Sentence -> Word/Token

    - Feature groups:
        - POS
        - Lexical
        - Dependency

    - Lexical features:
        - word especially which are mostly unique property for a entity class e.g. temperature
        - Begin of sentence
            - Also if conjunction with begin of sentence
        - Word mapped to multi-token
            - Handle their POS, dependency features
        - Numerals
            - 200 µL
            - 80%

    - Dependency features:
        - dependency type
        - dependent words
        - governor words
        - dependency consistency features as mentioned in Jie et al  papers/NLP/NER
"""
