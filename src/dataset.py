#!/usr/bin/env python

"""
Load dataset and the annotations.

Data Format:
-----------
http://noisy-text.github.io/2020/wlp-task.html#format

Each line of the protocol text indicates a single step in the protocol.
Observation: Step can have multiple sentences e.g. protocol_101 (line #2)
"""


import argparse
import codecs
import os
import re

from .annotation import *
from .nlp_process import *


class Line:
    """Protocol text line representing a step in protocol.
        A step consists of single or multiple sentences.
    """
    def __init__(self, start_char_pos=None, end_char_pos=None, start_word_index=None, end_word_index=None,
                 start_sent_index=None, end_sent_index=None):
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos
        self.start_word_index = start_word_index
        self.end_word_index = end_word_index
        self.start_sent_index = start_sent_index
        self.end_sent_index = end_sent_index


class Word:
    """Word class is associated with Line class."""
    def __init__(self, start_char_pos=None, end_char_pos=None):
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos


class Sentence:
    def __init__(self, start_char_pos=None, end_char_pos=None, start_token_index=None, end_token_index=None):
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index


class Token:
    """Token class is associated with Sentence class.
        Though this is similar to Word but created using different tokenizer(spaCy)
            which leads to difference in few words.
            Word represents token created by tokenizer which was used to named entities of train data.

        Additionally Token's tokenizer also extracts
            a) part of speech tagging
            b) syntactic parsing
    """
    def __init__(self, start_char_pos=None, end_char_pos=None, part_of_speech=None, dependency_tag=None, head_index=None, children_index_arr=None):
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos
        self.part_of_speech = part_of_speech
        self.dependency_tag = dependency_tag
        self.head_index = head_index
        self.children_index_arr = children_index_arr


class Document:
    """Protocol document"""
    def __init__(self, doc_id, nlp_process_obj):
        self.id = doc_id
        self.text = None

        # Line represents a protocol step. This may contain multiple sentences.
        self.lines = []
        self.words = []  # usually represents a space separated token in text.

        # Sentence represents a sentence in linguistic context.
        self.sentences = []
        # Token represents a token formed by NLP engine.
        self.tokens = []
        # Example: 37°C - This word is represented by 3 tokens in spaCy: 37, °, C

        self.entity_annotations = []

        self.nlp_process_obj = nlp_process_obj

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
            current_line_tags = []  # list of (word, NER tag) tuple
            char_pos = 0
            for line in fd:
                line = line.strip()

                if line == "":
                    # empty line represents protocol step change
                    if len(current_line_tags) > 0:
                        self.parse_protocol_step(current_line_tags=current_line_tags, char_pos=char_pos)

                        # update char_pos to end of latest added protocol step
                        char_pos = self.lines[-1].end_char_pos

                    # now empty the current_line_tags for the next protocol step
                    current_line_tags = []
                else:
                    tokens = line.split("\t")
                    current_line_tags.append(tuple(tokens))

            if len(current_line_tags) > 0:
                self.parse_protocol_step(current_line_tags=current_line_tags, char_pos=char_pos)

    def parse_protocol_step(self, current_line_tags, char_pos):
        """Parse protocol step to populate words, lines and entity annotations.

            Parameters:
            ----------
            current_line_tags : list of tuple (word, NER tag)
            char_pos : int (char position offset)
                        Current sentence text is searched from this offset onwards.
                        Usually its char position of end of the previous sentence.

            See Also:
            --------
            parse_conll_annotation
        """
        start_char_pos_line = None
        start_word_index = len(self.words)
        start_entity_ann_index = len(self.entity_annotations)

        for i, (word, ner_tag) in enumerate(current_line_tags):
            start_char_pos_word = char_pos + self.text[char_pos:].find(word)
            end_char_pos_word = start_char_pos_word + len(word)

            if i == 0:
                start_char_pos_line = start_char_pos_word

            if ner_tag != "O":
                split_index = ner_tag.find("-")
                assert split_index > 0, "Expected format B-TAG or I-TAG. NER tag: {}".format(ner_tag)
                if ner_tag[:split_index] == "B":
                    entity_ann = EntityAnnotation(start_word_index=len(self.words), end_word_index=len(self.words)+1, entity_type=ner_tag[split_index+1:])
                    self.entity_annotations.append(entity_ann)
                elif ner_tag[:split_index] == "I":
                    self.entity_annotations[-1].end_word_index = len(self.words)+1
                else:
                    assert False, "Expected format B-TAG or I-TAG. NER tag: {}".format(ner_tag)

            # append to word list
            self.words.append(Word(start_char_pos=start_char_pos_word, end_char_pos=end_char_pos_word))

            # update char position to the end of current word
            char_pos = end_char_pos_word

        # store the count of sentences till now
        n_sents_upto_prev_line = len(self.sentences)

        # Now populate sentences and tokens
        end_char_pos_line = char_pos
        # reset char position to start position of protocol step
        char_pos = start_char_pos_line

        # Segment protocol step into sentence(s)
        sents = self.nlp_process_obj.sent_tokenize(text=self.text[start_char_pos_line: end_char_pos_line])

        for sent in sents:
            # sent type: Span (https://spacy.io/api/span)
            sent_text = sent.text
            start_char_pos_relative = self.text[char_pos:].find(sent_text)
            assert start_char_pos_relative >= 0,\
                "Sentence not found from char position: {} onwards. Sentence text: {}".format(char_pos, sent_text)
            start_char_pos_sent = char_pos + start_char_pos_relative
            end_char_pos_sent = start_char_pos_sent + len(sent_text)

            doc_sentence = self.nlp_process_obj.construct_doc(text=sent_text)
            n_tokens_upto_prev_sent = len(self.tokens)

            for token in doc_sentence:
                start_char_pos_token = char_pos + self.text[char_pos:].find(token.text)
                end_char_pos_token = start_char_pos_token + len(token.text)

                head_index_token = token.head.i + n_tokens_upto_prev_sent
                children_index_arr_token = [(child.i + n_tokens_upto_prev_sent) for child in token.children]

                cur_token = Token(start_char_pos=start_char_pos_token, end_char_pos=end_char_pos_token,
                                  part_of_speech=token.pos_, dependency_tag=token.dep_, head_index=head_index_token,
                                  children_index_arr=children_index_arr_token)
                self.tokens.append(cur_token)

                # update char position to end of current token
                char_pos = start_char_pos_token

            # append sentence to sentence list
            self.sentences.append(Sentence(start_char_pos=start_char_pos_sent, end_char_pos=end_char_pos_sent,
                                           start_token_index=n_tokens_upto_prev_sent, end_token_index=len(self.tokens)))

            # update char position to the end of current sentence
            char_pos = end_char_pos_sent

        # append line to line list
        self.lines.append(Line(start_char_pos=start_char_pos_line, end_char_pos=char_pos,
                               start_word_index=start_word_index, end_word_index=len(self.words),
                               start_sent_index=n_sents_upto_prev_line, end_sent_index=len(self.sentences)))

    def display_document(self):
        ann_i = 0
        for line_i in range(len(self.lines)):
            start_char_pos_line = self.lines[line_i].start_char_pos
            end_char_pos_line = self.lines[line_i].end_char_pos
            start_sent_index_line = self.lines[line_i].start_sent_index
            end_sent_index_line = self.lines[line_i].end_sent_index

            line_text = self.text[start_char_pos_line: end_char_pos_line]

            print("\n\nLine #{} :: char pos range: ({}, {}) :: sent range: ({}, {}) :: text: {}".format(
                line_i, start_char_pos_line, end_char_pos_line, start_sent_index_line, end_sent_index_line, line_text))

            start_word_index_sent = self.lines[line_i].start_word_index
            end_word_index_sent = self.lines[line_i].end_word_index

            for word_index in range(start_word_index_sent, end_word_index_sent):
                word_text = self.text[self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]
                print("\tword #{}: {}".format(word_index, word_text))

            for sent_index in range(start_sent_index_line, end_sent_index_line):
                start_char_pos_sent = self.sentences[sent_index].start_char_pos
                end_char_pos_sent = self.sentences[sent_index].end_char_pos
                sent_text = self.text[start_char_pos_sent: end_char_pos_sent]

                print("\nSentence #{} :: char pos range: ({}, {}) :: text: {}".format(
                    sent_index, start_char_pos_sent, end_char_pos_sent, sent_text))

                start_token_index_sent = self.sentences[sent_index].start_token_index
                end_token_index_sent = self.sentences[sent_index].end_token_index

                for token_index in range(start_token_index_sent, end_token_index_sent):
                    start_char_pos_token = self.tokens[token_index].start_char_pos
                    end_char_pos_token = self.tokens[token_index].end_char_pos
                    token_text = self.text[start_char_pos_token: end_char_pos_token]
                    token_part_of_speech = self.tokens[token_index].part_of_speech
                    token_dependency_tag = self.tokens[token_index].dependency_tag
                    head_index_token = self.tokens[token_index].head_index

                    print("\tToken #{} :: char pos range: ({}, {}) :: text: {} :: POS: {} :: Dependency: {} :: "
                          "Head:  {}".format(
                        token_index, start_char_pos_token, end_char_pos_token, token_text, token_part_of_speech,
                        token_dependency_tag, head_index_token))

            # Find the entity annotations belonging to the line
            ann_index_sent_arr = []

            while ann_i < len(self.entity_annotations):
                start_word_index_ann = self.entity_annotations[ann_i].start_word_index
                end_word_index_ann = self.entity_annotations[ann_i].end_word_index

                assert start_word_index_ann >= start_word_index_sent,\
                    "annotation assignment to line missed by previous line. ann_i: {}".format(ann_i)

                if start_word_index_ann < end_word_index_sent:
                    ann_index_sent_arr.append(ann_i)
                else:
                    break

                ann_i += 1

            if len(ann_index_sent_arr) > 0:
                print("\nEntity annotations belonging to the step:")
                for ann_index in ann_index_sent_arr:
                    start_word_index_ann = self.entity_annotations[ann_index].start_word_index
                    end_word_index_ann = self.entity_annotations[ann_index].end_word_index
                    entity_type_ann = self.entity_annotations[ann_index].type

                    entity_text = " ".join(
                        [self.text[self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]
                         for word_index in range(start_word_index_ann, end_word_index_ann)])

                    print("\tEntity annotation #{} :: word index range: ({}, {}) :: type: {} :: text: {}".format(
                        ann_index, start_word_index_ann, end_word_index_ann, entity_type_ann, entity_text))


        print("\n\nEntity Annotations:")
        for ann_i in range(len(self.entity_annotations)):
            start_word_index_ann = self.entity_annotations[ann_i].start_word_index
            end_word_index_ann = self.entity_annotations[ann_i].end_word_index
            entity_type_ann = self.entity_annotations[ann_i].type

            entity_text = " ".join([self.text[self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]
                                    for word_index in range(start_word_index_ann, end_word_index_ann)])

            print("\tEntity annotation #{} :: word index range: ({}, {}) :: type: {} :: text: {}".format(
                ann_i, start_word_index_ann, end_word_index_ann, entity_type_ann, entity_text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol_id", action="store", dest="protocol_id")
    parser.add_argument("--data_dir", action="store", default="C:/KA/lib/WNUT_2020/data/train_data/", dest="data_dir")

    args = parser.parse_args()

    file_document = os.path.join(args.data_dir, "Standoff_Format/protocol_" + args.protocol_id + ".txt")
    file_conll_ann = os.path.join(args.data_dir, "Conll_Format/protocol_" + args.protocol_id + "_conll.txt")

    obj_nlp_process = NLPProcess()
    obj_nlp_process.load_nlp_model(verbose=True)
    obj_nlp_process.build_sentencizer(verbose=True)

    document_obj = Document(doc_id=int(args.protocol_id), nlp_process_obj=obj_nlp_process)
    document_obj.parse_document(document_file=file_document)
    document_obj.parse_conll_annotation(conll_ann_file=file_conll_ann)
    document_obj.display_document()

