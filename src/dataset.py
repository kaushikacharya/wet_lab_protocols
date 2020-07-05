"""
Load dataset and the annotations.
"""

import codecs
import os
import re

class Document:
    def __init__(self):
        self.text = None
        self.sentences = []
        self.words = []

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
            current_sentence_tags = []
            char_pos = 0
            for line in fd:
                line = line.strip()
                if line == "":
                    # empty line represents sentence change
                    if len(current_sentence_tags) > 0:
                        self.populate_sentence(current_sentence_tags=current_sentence_tags, char_pos=char_pos)
                    # now empty current_sentence_tags for the next sentence
                    current_sentence_tags = []
                    continue

                tokens = line.split("\t")
                current_sentence_tags.append(tuple(tokens))

            if len(current_sentence_tags) > 0:
                self.populate_sentence(current_sentence_tags=current_sentence_tags, char_pos=char_pos)

    def populate_sentence(self, current_sentence_tags, char_pos):
        start_char_pos_sent = None
        start_word_index = len(self.words)
        for i, (word, ner_tag) in enumerate(current_sentence_tags):
            start_char_pos_word = char_pos + self.text[char_pos:].find(word)
            end_char_pos_word = start_char_pos_word + len(word)
            if i == 0:
                start_char_pos_sent = start_char_pos_word
            self.words.append(Word(start_char_pos=start_char_pos_word, end_char_pos=end_char_pos_word))

            # update char position to the end of current word
            char_pos = end_char_pos_word

        self.sentences.append(Sentence(start_char_pos=start_char_pos_sent, end_char_pos=char_pos,
                                       start_word_index=start_word_index, end_word_index=len(self.words)))

    def display_document(self):
        for sent_i in range(len(self.sentences)):
            start_char_pos_sent = self.sentences[sent_i].start_char_pos
            end_char_pos_sent = self.sentences[sent_i].end_char_pos
            sentence_text = self.text[start_char_pos_sent: end_char_pos_sent]

            print("Sentence #{} :: char pos range: ({}, {}) :: text: {}".format(
                sent_i, start_char_pos_sent, end_char_pos_sent, sentence_text))

            start_word_index = self.sentences[sent_i].start_word_index
            end_word_index = self.sentences[sent_i].end_word_index

            for word_index in range(start_word_index, end_word_index):
                word_text = self.text[self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]
                print("\tword #{}: {}".format(word_index, word_text))

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
    document_obj = Document()
    document_obj.parse_document(document_file=file_document)
    document_obj.parse_conll_annotation(conll_ann_file=file_conll_ann)
    document_obj.display_document()

