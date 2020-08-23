#!/usr/bin/env python

"""
Load dataset and the annotations.

Data Format
-----------
    http://noisy-text.github.io/2020/wlp-task.html#format

    Each line of the protocol text indicates a single step in the protocol.
    Observation: Step can have multiple sentences e.g. protocol_101 (line #2)

Word Tokenizers
---------------
    Following two tokenizers have been used:

    1. Penn Treebank Tokenizer
        "Word" class represents the tokens formed by this tokenizer.
        Named Entity labels are mapped to Word.
    2. spaCy's word tokenizer
        "Token" class represents the tokens formed by this tokenizer.
        Syntactic features: part-of-speech tagging, dependency parse are associated with these tokens.

    Mapping of "Word" to "Token" class are done based on the text span.

"""


import argparse
import codecs
import nltk
import re

from .annotation import *


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
    def __init__(self, start_char_pos=None, end_char_pos=None, start_token_index=None, end_token_index=None, named_entity_label=None):
        """

        :param start_char_pos:
        :param end_char_pos:
        :param start_token_index:
        :param end_token_index:
        :param named_entity_label: str
                            BIO format labeling for train data
                            None for test data
        """
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos
        # mapping word to token
        self.start_token_index = start_token_index
        # Though usually end_token_index can be extracted from next word's start_token_index, but there are exceptions
        # in which tokens are formed in space between the words. e.g. protocol #3: Line #9
        self.end_token_index = end_token_index
        self.named_entity_label = named_entity_label


class Sentence:
    """Represents sentence in a linguistic sense."""
    def __init__(self, start_char_pos=None, end_char_pos=None, start_token_index=None, end_token_index=None,
                 start_word_index=None, end_word_index=None):
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos
        self.start_token_index = start_token_index
        self.end_token_index = end_token_index
        # word index range
        # Though Word is associated with Line, but storing word index range is required for feature extraction.
        self.start_word_index = start_word_index
        self.end_word_index = end_word_index


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
        """Assumption:
            Lists are stored in sequential order i.e. i'th element comes before elements at position (i+1)'th onwards.
        """
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
            with open(document_file, mode="r", encoding="utf-8") as fd:
                text = fd.read()

            # remove carriage return as annotations are based on unix style line ending
            self.text = re.sub(r'\r\n', '\n', text)
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
                        self.parse_protocol_step_conll(current_line_tags=current_line_tags, char_pos=char_pos, verbose=verbose)

                        # update char_pos to end of latest added protocol step
                        char_pos = self.lines[-1].end_char_pos

                    # now empty the current_line_tags for the next protocol step
                    current_line_tags = []
                else:
                    tokens = line.split("\t")
                    current_line_tags.append(tuple(tokens))

            if len(current_line_tags) > 0:
                self.parse_protocol_step_conll(current_line_tags=current_line_tags, char_pos=char_pos, verbose=verbose)

    def parse_standoff_annotation(self, ann_file, verbose=False):
        """Parse brat standoff formatted annotation file.

            Reference
            ---------
            https://brat.nlplab.org/standoff.html
        """
        assert self.text is not None, "parse_document() is a pre-requisite"
        # read the standoff annotations for the protocol
        with codecs.open(filename=ann_file, mode="r", encoding="utf-8") as fd:
            ann_tuple_arr = []
            for line in fd:
                line = line.strip()

                if line == "":
                    continue

                # TODO Currently only text bound annotations are considered as focus on only named entities and ignoring the relations.
                if line[0] != "T":
                    continue

                tokens = line.split("\t")
                assert len(tokens) == 3, "Expected three tab separated tokens for line: {}".format(line)

                entity_id = tokens[0]
                char_index = tokens[1].find(" ")
                assert char_index > 0, "space not found in middle token :: line: {}".format(line)
                entity_type = tokens[1][:char_index]
                entity_char_pos_arr = [int(x) for x in re.findall(r'\d+', tokens[1])]

                ann_tuple_arr.append((entity_id, entity_type, entity_char_pos_arr))

            # Now sort ann_tuple_arr wrt start character offset
            ann_tuple_arr = sorted(ann_tuple_arr, key=lambda x: x[2][0])

            if verbose:
                print("{}".format([(x[0], x[2]) for x in ann_tuple_arr]))


        # Now populate words, lines, sentences and map words to entity annotations
        if False:
            char_pos = 0
            for m in re.finditer(r'\n+', self.text):
                print(self.text[char_pos:m.start()])
                char_pos = m.end()

            # Handling the portion of the text(if available) after the final newline
            if char_pos < len(self.text):
                print(self.text[char_pos:])

        # iterate over the protocol steps (represented by line of text)
        char_pos = 0
        for m in re.finditer(r'\n+', self.text):
            self.parse_protocol_step_standoff(start_char_pos_line=char_pos, end_char_pos_line=m.start(), verbose=verbose)
            # update char_pos to start of next line
            char_pos = m.end()

        # Processing the text after the final newline, if available
        if char_pos < len(self.text):
            self.parse_protocol_step_standoff(start_char_pos_line=char_pos, end_char_pos_line=len(self.text), verbose=verbose)

    def parse_protocol_step_conll(self, current_line_tags, char_pos, verbose=False):
        """Parse protocol step to populate words, lines and entity annotations.
            This uses CoNLL annotations.

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
        start_word_index_line = len(self.words)
        start_entity_ann_index = len(self.entity_annotations)

        for i, (word, ner_tag) in enumerate(current_line_tags):
            if word == "``" or word == "''":
                # case: Double quotes changed by Penn Treebank Tokenizer
                # https://stackoverflow.com/questions/31074682/nltk-word-tokenize-changes-quotes/63334845
                start_char_pos_relative = self.text[char_pos:].find('"')
                word_length = 1
            else:
                start_char_pos_relative = self.text[char_pos:].find(word)
                word_length = len(word)

            assert start_char_pos_relative >= 0, "word: {} not found".format(word.encode("utf-8"))
            start_char_pos_word = char_pos + start_char_pos_relative
            end_char_pos_word = start_char_pos_word + word_length

            if i == 0:
                start_char_pos_line = start_char_pos_word

            if ner_tag is not None and ner_tag != "O":
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
            self.words.append(Word(start_char_pos=start_char_pos_word, end_char_pos=end_char_pos_word, named_entity_label=ner_tag))

            # update char position to the end of current word
            char_pos = end_char_pos_word

        # store the count of sentences till now
        n_sents_upto_prev_line = len(self.sentences)

        # Now populate sentences and tokens
        end_char_pos_line = char_pos
        # reset char position to start position of protocol step
        char_pos = start_char_pos_line

        if verbose:
            line_text = self.text[start_char_pos_line: end_char_pos_line]
            print("-----------------------------------------------------------")
            print("\nLine: {}\n".format(line_text.encode("utf-8")))

        # Segment protocol step into sentence(s)
        sents = self.nlp_process_obj.sent_tokenize(text=self.text[start_char_pos_line: end_char_pos_line])

        # initialize word_index to the first word of the line i.e. protocol step
        word_index = start_word_index_line
        # As we iterate over the tokens of the sentence to map with the words of the line,
        #  ensure that word_index stays within the line.

        for sent in sents:
            # sent type: Span (https://spacy.io/api/span)
            sent_text = sent.text

            if verbose:
                print("\nSentence: {}".format(sent_text.encode("utf-8")))

            # Compute char position range for the sentence
            start_char_pos_relative = self.text[char_pos:].find(sent_text)
            assert start_char_pos_relative >= 0,\
                "Sentence not found from char position: {} onwards. Sentence text: {}".format(char_pos, sent_text.encode("utf-8"))
            start_char_pos_sent = char_pos + start_char_pos_relative
            end_char_pos_sent = start_char_pos_sent + len(sent_text)

            start_word_index_sent = word_index

            doc_sentence = self.nlp_process_obj.construct_doc(text=sent_text)
            n_tokens_upto_prev_sent = len(self.tokens)

            for token in doc_sentence:
                start_char_pos_token = char_pos + self.text[char_pos:].find(token.text)
                end_char_pos_token = start_char_pos_token + len(token.text)

                if verbose:
                    word_text = self.text[self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]
                    print("\ttoken: {} :: char pos range(token): ({}, {}) :: word: {} :: word_index: {} :: char pos range(word): ({}, {})".format(
                        token.text.encode("utf-8"), start_char_pos_token, end_char_pos_token, word_text.encode("utf-8"),
                        word_index, self.words[word_index].start_char_pos, self.words[word_index].end_char_pos))

                head_index_token = token.head.i + n_tokens_upto_prev_sent
                children_index_arr_token = [(child.i + n_tokens_upto_prev_sent) for child in token.children]

                # ----- Map token(s) to the word(s) --------

                # Usually token to word map is one-to-one.
                # Exceptions: a) many-to-one e.g. word: 37°C  tokens: [37, °, C]
                #             b) one-to-many e.g. token: node?The  words: [node, ?, The] (source: protocol_101  line #23)
                # Exception (b) is rare.

                if end_char_pos_token <= self.words[word_index].start_char_pos:
                    # case: token represents space between the current(represented by word_index) and previous word
                    #       Since this token is formed between words, hence not mapped to any word.
                    #       Rare instance. Happens when a token is formed in the space between words.
                    pass
                elif start_char_pos_token >= self.words[word_index].end_char_pos:
                    # Following assert ensures that token is entirely beyond the word represented by word_index
                    # only when token is formed by the space beyond the last word of the protocol step.
                    assert word_index == (len(self.words)-1), "token not expected beyond the word_index: {}".format(word_index)
                else:
                    # case: token maps to the word represented by word_index

                    # update word to token map
                    if self.words[word_index].start_token_index is None:
                        # case: First token belonging to the word
                        self.words[word_index].start_token_index = len(self.tokens)

                    # update the end_token_index
                    self.words[word_index].end_token_index = len(self.tokens) + 1

                    # Now look into the different sub-cases
                    if end_char_pos_token == self.words[word_index].end_char_pos:
                        # case: token's end is same as word's (represented by word_index) end
                        # increment word_index if its not the last word of this line
                        # increment ensures that next token is matched with updated word_index's word
                        if word_index < (len(self.words)-1):
                            word_index += 1
                    elif end_char_pos_token < self.words[word_index].end_char_pos:
                        # case: multiple tokens split from the word
                        # Next token(s) expected to belong to the word. Hence word_index not incremented.
                        pass
                    else:
                        # case: More word(s) map to the current token
                        while word_index < (len(self.words)-1) and (self.words[word_index+1].start_char_pos < end_char_pos_token):
                            word_index += 1
                            # update word to token map
                            self.words[word_index].start_token_index = len(self.tokens)
                            self.words[word_index].end_token_index = len(self.tokens) + 1

                        # increment word_index if its not the last word of this line
                        if word_index < (len(self.words)-1):
                            word_index += 1

                '''
                # Usually token should either map to word represented by word_index or to the next word
                # Exception: token is formed from the space between words. In this case, skip the mapping.
                if end_char_pos_token <= self.words[word_index].start_char_pos:
                    # case: token represents space before the first word of the sentence
                    pass
                elif start_char_pos_token < self.words[word_index].end_char_pos:
                    # case: token maps to the word represented by word_index
                    if self.words[word_index].start_token_index is None:
                        # case: First token belonging to the word
                        self.words[word_index].start_token_index = len(self.tokens)

                    # update the end_token_index
                    self.words[word_index].end_token_index = len(self.tokens) + 1
                else:
                    # Now the token needs to be checked if it belongs to the next word(if available)

                    # case a: token maps to the space after the word represented by word_index
                    # case b: token maps to the next word
                    if word_index < len(self.words)-1:
                        word_index += 1

                        token_text = self.text[start_char_pos_token: end_char_pos_token]
                        word_text = self.text[self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]

                        assert start_char_pos_token < self.words[word_index].end_char_pos,\
                            "No corresponding token for the word_index: {} :: word: {} :: token: {} :: " \
                            "char pos range: (word): ({},{}) : (token): ({},{}) ".format(
                                word_index, word_text.encode("utf-8"), token_text.encode("utf-8"),
                                self.words[word_index].start_char_pos, self.words[word_index].end_char_pos,
                                start_char_pos_token, end_char_pos_token
                            )

                        if start_char_pos_token >= self.words[word_index].start_char_pos:
                            self.words[word_index].start_token_index = len(self.tokens)
                            self.words[word_index].end_token_index = len(self.tokens) + 1
                            if verbose:
                                word_text = self.text[
                                            self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]
                                print("\t\tToken mapped to word_index: {} :: word: {}".format(word_index, word_text.encode("utf-8")))

                if (word_index < len(self.words)) and (start_char_pos_token >= self.words[word_index].end_char_pos):
                    # token maps to next word
                    word_index += 1
                '''

                cur_token = Token(start_char_pos=start_char_pos_token, end_char_pos=end_char_pos_token,
                                  part_of_speech=token.pos_, dependency_tag=token.dep_, head_index=head_index_token,
                                  children_index_arr=children_index_arr_token)
                self.tokens.append(cur_token)

                # update char position to end of current token
                char_pos = end_char_pos_token

            # set end word index for the sentence
            # Current word_index could either point to the last word of the sentence or the first word of the next sentence.
            if self.words[word_index].start_char_pos >= end_char_pos_sent:
                end_word_index_sent = word_index
            else:
                end_word_index_sent = word_index + 1

            # append sentence to sentence list
            self.sentences.append(Sentence(start_char_pos=start_char_pos_sent, end_char_pos=end_char_pos_sent,
                                           start_token_index=n_tokens_upto_prev_sent, end_token_index=len(self.tokens),
                                           start_word_index=start_word_index_sent, end_word_index=end_word_index_sent))

            # update char position to the end of current sentence
            char_pos = end_char_pos_sent

        # append line to line list
        self.lines.append(Line(start_char_pos=start_char_pos_line, end_char_pos=char_pos,
                               start_word_index=start_word_index_line, end_word_index=len(self.words),
                               start_sent_index=n_sents_upto_prev_line, end_sent_index=len(self.sentences)))

    def parse_protocol_step_standoff(self, start_char_pos_line, end_char_pos_line, verbose=False):
        """Parse protocol step using Standoff format annotations.
            Populate sentences, words, tokens for the protocol step.

            Parameters
            ----------
            start_char_pos_line : int
                character offset representing start of the protocol step text
            end_char_pos_line : int
                character offset representing end of the protocol step text
            verbose : bool
        """

        if verbose:
            line_text = self.text[start_char_pos_line: end_char_pos_line]
            print("\nLine #{} :: char pos range: ({}, {}) :: text: {}".format(len(self.lines), start_char_pos_line,
                                                                              end_char_pos_line, line_text.encode("utf-8")))

        # store the count of sentences till now
        start_sent_index_line = len(self.sentences)

        # Segment protocol step text into sentence(s)
        sents = self.nlp_process_obj.sent_tokenize(text=self.text[start_char_pos_line: end_char_pos_line])

        char_pos = start_char_pos_line

        for sent in sents:
            # sent type: Span (https://spacy.io/api/span)
            sent_text = sent.text

            # Compute char position range for the sentence
            start_char_pos_relative = self.text[char_pos:].find(sent_text)
            assert start_char_pos_relative >= 0,\
                "Sentence not found from char position: {} onwards. Sentence text: {}".format(
                    char_pos, sent_text.encode("utf-8"))
            start_char_pos_sent = char_pos + start_char_pos_relative
            end_char_pos_sent = start_char_pos_sent + len(sent_text)

            if verbose:
                print("\n\tSentence #{}: char position span:({}, {}) :: text: {}".format(
                    len(self.sentences), start_char_pos_sent, end_char_pos_sent, sent_text.encode("utf-8")))

            # -------- Populate tokens of the current sentence ----------
            # These tokens are created by spaCy pipeline.
            doc_sentence = self.nlp_process_obj.construct_doc(text=sent_text)

            start_token_index_sent = len(self.tokens)
            for token in doc_sentence:
                # character position span of the token
                start_char_pos_token = char_pos + self.text[char_pos:].find(token.text)
                end_char_pos_token = start_char_pos_token + len(token.text)

                head_index_token = token.head.i + start_token_index_sent
                children_index_arr_token = [(child.i + start_token_index_sent) for child in token.children]

                if verbose:
                    token_text = self.text[start_char_pos_token: end_char_pos_token]
                    print("\t\tToken #{} :: char pos range: ({}, {}) :: text: {} :: POS: {} :: Dependency: {} ::"
                          " Head: {}".format(len(self.tokens), start_char_pos_token, end_char_pos_token,
                                             token_text.encode("utf-8"), token.pos_, token.dep_, head_index_token))

                cur_token = Token(start_char_pos=start_char_pos_token, end_char_pos=end_char_pos_token,
                                  part_of_speech=token.pos_, dependency_tag=token.dep_, head_index=head_index_token,
                                  children_index_arr=children_index_arr_token)
                self.tokens.append(cur_token)

                # update char position to end of current token
                char_pos = end_char_pos_token

            # --------- Populate words of the current sentence -----------
            word_tokens = nltk.word_tokenize(text=sent_text)

            # set char position to the start of the sentence as it was moved ahead in the for loop for tokens
            char_pos = start_char_pos_sent

            start_word_index_sent = len(self.words)
            token_index = start_token_index_sent

            for word in word_tokens:
                if word == "``" or word == "''":
                    # case: Double quotes changed by Penn Treebank Tokenizer
                    # https://stackoverflow.com/questions/31074682/nltk-word-tokenize-changes-quotes/63334845
                    start_char_pos_relative = self.text[char_pos:].find('"')
                    word_length = 1
                else:
                    start_char_pos_relative = self.text[char_pos:].find(word)
                    word_length = len(word)

                assert start_char_pos_relative >= 0, "word: {} not found".format(word.encode("utf-8"))
                start_char_pos_word = char_pos + start_char_pos_relative
                end_char_pos_word = start_char_pos_word + word_length

                # ------ Map tokens overlapping with the current word ------
                # case: Handling tokens formed in the space between words
                while self.tokens[token_index].end_char_pos <= start_char_pos_word:
                    token_index += 1
                    assert token_index < len(self.tokens),\
                        "no more tokens left, but still these are before the word represented by word_index: {}".format(len(self.words))

                assert token_index < len(self.tokens), "token_index: {} moved beyond total tokens: {}".format(token_index, len(self.tokens))
                assert self.tokens[token_index].start_char_pos < end_char_pos_word,\
                    "No token matched with word(represented by word_index: {}) :: token_index: {}".format(len(self.words), token_index)

                start_token_index_word = token_index

                # iterate over the tokens which overlap with the current word
                while (token_index < len(self.tokens)) and (self.tokens[token_index].end_char_pos < end_char_pos_word):
                    token_index += 1

                end_token_index_word = token_index + 1

                if verbose:
                    word_text = self.text[start_char_pos_word: end_char_pos_word]
                    print("\t\tWord #{}: {} :: char pos range: ({}, {}) :: token range: ({}, {})".format(
                        len(self.words), word_text.encode("utf-8"), start_char_pos_word, end_char_pos_word,
                        start_token_index_word, end_token_index_word))

                # append to word list
                self.words.append(Word(start_char_pos=start_char_pos_word, end_char_pos=end_char_pos_word,
                                       start_token_index=start_token_index_word, end_token_index=end_token_index_word))

                # update char position to the end of current word
                char_pos = end_char_pos_word

            # append sentence to sentence list
            self.sentences.append(Sentence(start_char_pos=start_char_pos_sent, end_char_pos=end_char_pos_sent,
                                           start_token_index=start_token_index_sent, end_token_index=len(self.tokens),
                                           start_word_index=start_word_index_sent, end_word_index=len(self.words)))

            # update char position to the end of current sentence
            char_pos = end_char_pos_sent

        # append line to the list
        self.lines.append(Line(start_char_pos=start_char_pos_line, end_char_pos=end_char_pos_line,
                               start_sent_index=start_sent_index_line, end_sent_index=len(self.sentences)))

    def display_document(self):
        ann_i = 0
        for line_i in range(len(self.lines)):
            start_char_pos_line = self.lines[line_i].start_char_pos
            end_char_pos_line = self.lines[line_i].end_char_pos
            start_sent_index_line = self.lines[line_i].start_sent_index
            end_sent_index_line = self.lines[line_i].end_sent_index

            line_text = self.text[start_char_pos_line: end_char_pos_line]

            print("\n\nLine #{} :: char pos range: ({}, {}) :: sent range: ({}, {}) :: text: {}".format(
                line_i, start_char_pos_line, end_char_pos_line, start_sent_index_line, end_sent_index_line,
                line_text.encode("utf-8")))

            start_word_index_line = self.lines[line_i].start_word_index
            end_word_index_line = self.lines[line_i].end_word_index

            for word_index in range(start_word_index_line, end_word_index_line):
                start_char_pos_word = self.words[word_index].start_char_pos
                end_char_pos_word = self.words[word_index].end_char_pos
                word_text = self.text[start_char_pos_word: end_char_pos_word]
                start_token_index = self.words[word_index].start_token_index
                end_token_index = self.words[word_index].end_token_index
                named_entity_label = self.words[word_index].named_entity_label
                print("\tword #{}: {} :: char pos range: ({}, {}) :: token range: ({}, {}) :: NER label: {}".format(
                    word_index, word_text.encode("utf-8"), start_char_pos_word, end_char_pos_word, start_token_index,
                    end_token_index, named_entity_label))

            for sent_index in range(start_sent_index_line, end_sent_index_line):
                start_char_pos_sent = self.sentences[sent_index].start_char_pos
                end_char_pos_sent = self.sentences[sent_index].end_char_pos
                start_word_index_sent = self.sentences[sent_index].start_word_index
                end_word_index_sent = self.sentences[sent_index].end_word_index
                sent_text = self.text[start_char_pos_sent: end_char_pos_sent]

                print("\nSentence #{} :: char pos range: ({}, {}) :: word index range: ({}, {}) :: text: {}".format(
                    sent_index, start_char_pos_sent, end_char_pos_sent, start_word_index_sent, end_word_index_sent,
                    sent_text.encode("utf-8")))

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
                          "Head: {}".format(token_index, start_char_pos_token, end_char_pos_token,
                                            token_text.encode("utf-8"), token_part_of_speech,
                                            token_dependency_tag, head_index_token))

            # Find the entity annotations belonging to the line
            ann_index_sent_arr = []

            while ann_i < len(self.entity_annotations):
                start_word_index_ann = self.entity_annotations[ann_i].start_word_index
                end_word_index_ann = self.entity_annotations[ann_i].end_word_index

                assert start_word_index_ann >= start_word_index_line,\
                    "annotation assignment to line missed by previous line. ann_i: {} :: " \
                    "word index range(ann): ({}, {})".format(ann_i, start_word_index_ann, end_word_index_ann)

                if start_word_index_ann < end_word_index_line:
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
                        ann_index, start_word_index_ann, end_word_index_ann, entity_type_ann, entity_text.encode("utf-8")))


        print("\n\nEntity Annotations:")
        for ann_i in range(len(self.entity_annotations)):
            start_word_index_ann = self.entity_annotations[ann_i].start_word_index
            end_word_index_ann = self.entity_annotations[ann_i].end_word_index
            entity_type_ann = self.entity_annotations[ann_i].type

            entity_text = " ".join([self.text[self.words[word_index].start_char_pos: self.words[word_index].end_char_pos]
                                    for word_index in range(start_word_index_ann, end_word_index_ann)])

            print("\tEntity annotation #{} :: word index range: ({}, {}) :: type: {} :: text: {}".format(
                ann_i, start_word_index_ann, end_word_index_ann, entity_type_ann, entity_text.encode("utf-8")))

def main(args):
    import os
    from .nlp_process import NLPProcess

    file_document = os.path.join(args.data_dir, "Standoff_Format/protocol_" + args.protocol_id + ".txt")
    if args.ann_format == "conll":
        file_ann = os.path.join(args.data_dir, "Conll_Format/protocol_" + args.protocol_id + "_conll.txt")
    elif args.ann_format == "standoff":
        file_ann = os.path.join(args.data_dir, "Standoff_Format/protocol_" + args.protocol_id + ".ann")
    else:
        assert False, "Expected ann_format: a) conll,  b) standoff. Received: {}".format(args.ann_format)

    obj_nlp_process = NLPProcess(model=args.model)
    obj_nlp_process.load_nlp_model(verbose=True)
    obj_nlp_process.build_sentencizer(verbose=True)

    document_obj = Document(doc_id=int(args.protocol_id), nlp_process_obj=obj_nlp_process)
    print("Document id: {}".format(document_obj.id))
    document_obj.parse_document(document_file=file_document)
    if args.ann_format == "conll":
        document_obj.parse_conll_annotation(conll_ann_file=file_ann, verbose=args.verbose)
        document_obj.display_document()
    else:
        document_obj.parse_standoff_annotation(ann_file=file_ann, verbose=args.verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol_id", action="store", dest="protocol_id")
    parser.add_argument("--data_dir", action="store", default="C:/KA/lib/WNUT_2020/data/train_data/", dest="data_dir")
    parser.add_argument("--model", action="store", default="en_core_web_sm", dest="model")
    parser.add_argument("--ann_format", action="store", default="conll", dest="ann_format", help="Either conll or standoff")  #noqa
    parser.add_argument("--verbose", action="store_true", default=False, dest="verbose")

    args = parser.parse_args()
    main(args=args)
