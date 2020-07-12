#!/usr/bin/env python

"""
Load dataset and the annotations.
"""


import codecs
import os
import re

from .annotation import *

class Document:
    def __init__(self, doc_id):
        self.id = doc_id
        self.text = None
        self.sentences = []
        self.words = []
        self.entity_annotations = []

    def parse_document(self, document_file):
        """Load the document text
        """
        try:
            with codecs.open(filename=document_file, mode="r", encoding="utf-8") as fd:
                text = fd.read()

            # remove carriage return as annotations are based on unix style line ending
            self.text = re.sub(r'\r', '', text)
        except Exception as err:
            print(err)

    def parse_conll_annotation(self, conll_ann_file, verbose=False):
        assert self.text is not None, "parse_document() is a pre-requisite"
        with codecs.open(filename=conll_ann_file, mode="r", encoding="utf-8") as fd:
            current_sentence_tags = []  # list of (word, NER tag) tuple
            char_pos = 0
            for line in fd:
                line = line.strip()
                if line == "":
                    # empty line represents sentence change
                    if len(current_sentence_tags) > 0:
                        self.parse_sentence(current_sentence_tags=current_sentence_tags, char_pos=char_pos)

                        # update char_pos to end of latest added sentence
                        char_pos = self.sentences[-1].end_char_pos

                    # now empty current_sentence_tags for the next sentence
                    current_sentence_tags = []
                    continue

                tokens = line.split("\t")
                current_sentence_tags.append(tuple(tokens))

            if len(current_sentence_tags) > 0:
                self.parse_sentence(current_sentence_tags=current_sentence_tags, char_pos=char_pos)

    def parse_sentence(self, current_sentence_tags, char_pos):
        """Parse sentence

            Parameters:
            ----------
            current_sentence_tags : list of tuple (word, NER tag)
            char_pos : int (char position offset)
                        Current sentence text is searched from this offset onwards.
                        Usually its char position of end of the previous sentence.
        """
        start_char_pos_sent = None
        start_word_index = len(self.words)
        start_entity_ann_index = len(self.entity_annotations)

        for i, (word, ner_tag) in enumerate(current_sentence_tags):
            start_char_pos_word = char_pos + self.text[char_pos:].find(word)
            end_char_pos_word = start_char_pos_word + len(word)

            if i == 0:
                start_char_pos_sent = start_char_pos_word

            if ner_tag != "O":
                tag_tokens = ner_tag.split("-")
                assert len(tag_tokens) == 2, "Expected format B-TAG or I-TAG. NER tag: {}".format(ner_tag)
                if tag_tokens[0] == "B":
                    entity_ann = EntityAnnotation(start_word_index=len(self.words), end_word_index=len(self.words)+1, entity_type=tag_tokens[1])
                    self.entity_annotations.append(entity_ann)
                elif tag_tokens[0] == "I":
                    self.entity_annotations[-1].end_word_index = len(self.words)+1
                else:
                    assert False, "Expected format B-TAG or I-TAG. NER tag: {}".format(ner_tag)

            # append to word list
            self.words.append(Word(start_char_pos=start_char_pos_word, end_char_pos=end_char_pos_word))

            # update char position to the end of current word
            char_pos = end_char_pos_word

        # append sentence to sentence list
        self.sentences.append(Sentence(start_char_pos=start_char_pos_sent, end_char_pos=char_pos,
                                       start_word_index=start_word_index, end_word_index=len(self.words)))

    def display_document(self):
        ann_i = 0
        for sent_i in range(len(self.sentences)):
            start_char_pos_sent = self.sentences[sent_i].start_char_pos
            end_char_pos_sent = self.sentences[sent_i].end_char_pos
            sentence_text = self.text[start_char_pos_sent: end_char_pos_sent]

            print("\n\nSentence #{} :: char pos range: ({}, {}) :: text: {}".format(
                sent_i, start_char_pos_sent, end_char_pos_sent, sentence_text))

            start_word_index_sent = self.sentences[sent_i].start_word_index
            end_word_index_sent = self.sentences[sent_i].end_word_index

            for word_index in range(start_word_index_sent, end_word_index_sent):
                word_text = self.text[self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]
                print("\tword #{}: {}".format(word_index, word_text))

            # Find the entity annotations belonging to the sentence
            ann_index_sent_arr = []

            while ann_i < len(self.entity_annotations):
                start_word_index_ann = self.entity_annotations[ann_i].start_word_index
                end_word_index_ann = self.entity_annotations[ann_i].end_word_index

                assert start_word_index_ann >= start_word_index_sent, "annotation assignment to sentence missed by previous sentence. ann_i: {}".format(ann_i)

                if start_word_index_ann < end_word_index_sent:
                    ann_index_sent_arr.append(ann_i)
                else:
                    break

                ann_i += 1

            if len(ann_index_sent_arr) > 0:
                print("\nEntity annotations belonging to the sentence:")
                for ann_index in ann_index_sent_arr:
                    start_word_index_ann = self.entity_annotations[ann_index].start_word_index
                    end_word_index_ann = self.entity_annotations[ann_index].end_word_index
                    entity_type_ann = self.entity_annotations[ann_index].type

                    entity_text = " ".join(
                        [self.text[self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]
                         for word_index in range(start_word_index_ann, end_word_index_ann)])

                    print("\tEntity annotation #{} :: word index range: ({}, {}) :: type: {} :: text: {}".format(
                        ann_index, start_word_index_ann, end_word_index_ann, entity_type_ann, entity_text))


        for ann_i in range(len(self.entity_annotations)):
            start_word_index_ann = self.entity_annotations[ann_i].start_word_index
            end_word_index_ann = self.entity_annotations[ann_i].end_word_index
            entity_type_ann = self.entity_annotations[ann_i].type

            entity_text = " ".join([self.text[self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]
                                    for word_index in range(start_word_index_ann, end_word_index_ann)])

            print("Entity annotation #{} :: word index range: ({}, {}) :: type: {} :: text: {}".format(
                ann_i, start_word_index_ann, end_word_index_ann, entity_type_ann, entity_text))

class Sentence:
    def __init__(self, start_char_pos=None, end_char_pos=None, start_word_index=None, end_word_index=None):
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos
        self.start_word_index = start_word_index
        self.end_word_index = end_word_index


class Word:
    def __init__(self, start_char_pos=None, end_char_pos=None):
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos

if __name__ == "__main__":
    protocol_id = 3
    data_folder = "C:/KA/lib/WNUT_2020/data/train_data/"
    file_document = os.path.join(data_folder, "Standoff_Format/protocol_" + str(protocol_id) + ".txt")
    file_conll_ann = os.path.join(data_folder, "Conll_Format/protocol_" + str(protocol_id) + "_conll.txt")
    document_obj = Document(doc_id=protocol_id)
    document_obj.parse_document(document_file=file_document)
    document_obj.parse_conll_annotation(conll_ann_file=file_conll_ann)
    document_obj.display_document()

