#!/usr/bin/env python


class EntityAnnotation:
    """By entity we define 'Action' as well as entities.
        This class is tied to a particular Document.
    """
    def __init__(self, entity_id=None, start_char_pos=None, end_char_pos=None, start_word_index=None, end_word_index=None, entity_type=None):
        # id: defined in .ann file. Its defined as 'T' followed by numeral.
        self.id = entity_id
        self.start_char_pos = start_char_pos
        self.end_char_pos = end_char_pos
        # word index range: [start_word_index, end_word_index)
        self.start_word_index = start_word_index
        self.end_word_index = end_word_index
        self.type = entity_type
