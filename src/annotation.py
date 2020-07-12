#!/usr/bin/env python


class EntityAnnotation:
    """By entity we define 'Action' as well as entities.
        This class is tied to a particular Document.
    """
    def __init__(self, entity_id=None, start_word_index=None, end_word_index=None, entity_type=None):
        # id: defined in .ann file. Its defined as 'T' followed by numeral.
        self.id = entity_id
        # word index range: [start_word_index, end_word_index)
        self.start_word_index = start_word_index
        self.end_word_index = end_word_index
        self.type = entity_type
