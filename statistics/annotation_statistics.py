#!/usr/bin/env python

"""
Statistics for the annotations.
"""

import argparse
from csv import writer
import io
import os
import pandas as pd
import re
import traceback

from src.dataset import *
from src.nlp_process import *


class AnnotationStatistics:
    def __init__(self):
        self.entity_statistics_dict = dict()
        self.nlp_process_obj = None

        self.build_nlp_process(verbose=True)

    def build_nlp_process(self, verbose=True):
        self.nlp_process_obj = NLPProcess()
        self.nlp_process_obj.load_nlp_model(verbose=verbose)
        self.nlp_process_obj.build_sentencizer(verbose=verbose)

    def process_dataset(self, data_dir):
        """Process protocols in the dataset"""
        # Iterate over each of the files in the folder (train, dev)
        for f in os.listdir(os.path.join(data_dir, "Conll_Format")):
            # print(f)
            m = re.search(r"\d+", f)
            if m is None:
                print("file: {} not in expected format".format(f))
                continue

            protocol_id = m.group(0)
            file_document = os.path.join(data_dir, "Standoff_Format/protocol_" + protocol_id + ".txt")
            file_conll_ann = os.path.join(data_dir, "Conll_Format/protocol_" + protocol_id + "_conll.txt")

            if not os.path.exists(file_document):
                print("{} missing".format(file_document))
                continue

            if not os.path.exists(file_conll_ann):
                print("{} missing".format(file_conll_ann))
                continue

            # print("text file: {}, conll annotated file: {}".format(file_document, file_conll_ann))
            try:
                self.process_protocol(doc_id=int(protocol_id), doc_file=file_document, conll_ann_file=file_conll_ann)
            except:
                print("Crashed in protocol_id: {}".format(protocol_id))
                traceback.print_exc()

    def process_protocol(self, doc_id, doc_file, conll_ann_file):
        """Process a single protocol document"""
        doc_obj = Document(doc_id=doc_id, nlp_process_obj=self.nlp_process_obj)
        doc_obj.parse_document(document_file=doc_file)
        doc_obj.parse_conll_annotation(conll_ann_file=conll_ann_file)

        ann_i = 0
        # ??? Isn't that the text statistics can also be collected by iterating over the entities

        # iterate over the lines of the protocol where each line represents protocol step
        for line_i in range(len(doc_obj.lines)):
            start_word_index_line = doc_obj.lines[line_i].start_word_index
            end_word_index_line = doc_obj.lines[line_i].end_word_index

            # Find the entity annotations belonging to the protocol step
            ann_index_line_arr = []

            while ann_i < len(doc_obj.entity_annotations):
                start_word_index_ann = doc_obj.entity_annotations[ann_i].start_word_index
                end_word_index_ann = doc_obj.entity_annotations[ann_i].end_word_index

                assert start_word_index_ann >= start_word_index_line,\
                    "annotation assignment to sentence missed by previous sentence. ann_i: {}".format(ann_i)

                if start_word_index_ann < end_word_index_line:
                    ann_index_line_arr.append(ann_i)
                else:
                    break

                ann_i += 1

            # Now populate statistics
            for ann_index in ann_index_line_arr:
                start_word_index_ann = doc_obj.entity_annotations[ann_index].start_word_index
                end_word_index_ann = doc_obj.entity_annotations[ann_index].end_word_index
                entity_type_ann = doc_obj.entity_annotations[ann_index].type

                entity_text = " ".join(
                    [doc_obj.text[doc_obj.words[word_index].start_char_pos: doc_obj.words[word_index].end_char_pos]
                     for word_index in range(start_word_index_ann, end_word_index_ann)])

                if entity_type_ann not in self.entity_statistics_dict:
                    self.entity_statistics_dict[entity_type_ann] = EntityStatistics(entity_tag=entity_type_ann)

                self.entity_statistics_dict[entity_type_ann].count += 1
                entity_text_lower = entity_text.lower()
                if entity_text_lower not in self.entity_statistics_dict[entity_type_ann].text_dict:
                    self.entity_statistics_dict[entity_type_ann].text_dict[entity_text_lower] = []

                self.entity_statistics_dict[entity_type_ann].text_dict[entity_text_lower].append(doc_id)

        token_i = 0
        n_tokens = len(doc_obj.tokens)
        for ann_index in range(len(doc_obj.entity_annotations)):
            start_word_index_ann = doc_obj.entity_annotations[ann_index].start_word_index
            end_word_index_ann = doc_obj.entity_annotations[ann_index].end_word_index
            entity_type_ann = doc_obj.entity_annotations[ann_index].type

            # --- extract the tokens in the current entity annotation ---
            start_char_pos_ann = doc_obj.words[start_word_index_ann].start_char_pos
            end_char_pos_ann = doc_obj.words[end_word_index_ann-1].end_char_pos
            # N.B. end_word_index representing end of word index range is excluding word indexes belonging to the entity.
            #       Hence char position end of the entity is chosen from the previous word index

            # skip the tokens which are prior(in terms of char position) to the current entity annotation
            while doc_obj.tokens[token_i].end_char_pos <= start_char_pos_ann:
                token_i += 1
                assert token_i < n_tokens,\
                    "failed to identify the starting token matching entity annotation #{} :: " \
                    "word index range: ({}, {}) :: char pos range: ({}, {})".format(
                        ann_index, start_word_index_ann, end_word_index_ann, start_char_pos_ann, end_char_pos_ann)

            start_token_index_ann = token_i

            # identify the end of the token range matching the current entity annotation
            # N.B. As per convention, end of the range excludes the annotation i.e. range: [)
            token_i += 1
            while (token_i < n_tokens) and (doc_obj.tokens[token_i].start_char_pos < end_char_pos_ann):
                token_i += 1

            end_token_index_ann = token_i

            # Select the dependency tags for the heads which are outside of the token range of the entity annotation
            dependency_tags = []
            for token_index in range(start_token_index_ann, end_token_index_ann):
                head_index = doc_obj.tokens[token_index].head_index
                if (head_index == token_index) or (head_index < start_token_index_ann) or (head_index >= end_token_index_ann):
                    dependency_tags.append(doc_obj.tokens[token_index].dependency_tag)

            dependency_tags_str = "_".join(dependency_tags)

            entity_type_ann in self.entity_statistics_dict,\
            "entity type: {} should have been already in entity_statistics_dict. Entity annotation #{}".format(entity_type_ann, ann_index)

            if dependency_tags_str not in self.entity_statistics_dict[entity_type_ann].dependency_dict:
                self.entity_statistics_dict[entity_type_ann].dependency_dict[dependency_tags_str] = []

            self.entity_statistics_dict[entity_type_ann].dependency_dict[dependency_tags_str].append(doc_id)


    def display_entity_tag_stats(self):
        for entity_tag in self.entity_statistics_dict:
            print("entity: {} :: count: {} :: unique text count: {}".format(
                entity_tag, self.entity_statistics_dict[entity_tag].count, len(self.entity_statistics_dict[entity_tag].text_dict)))

    def write_annotation_data(self):
        """Write annotation data collected over the dataset.
            This would be useful in analyzing the instances of entity tags.
        """
        csv_output = io.StringIO()
        csv_writer = writer(csv_output)

        for entity_tag in self.entity_statistics_dict:
            for entity_text_lower in self.entity_statistics_dict[entity_tag].text_dict:
                doc_ids = self.entity_statistics_dict[entity_tag].text_dict[entity_text_lower]
                csv_writer.writerow([entity_tag, entity_text_lower, len(doc_ids), doc_ids])

        csv_output.seek(0)

        df = pd.read_csv(filepath_or_buffer=csv_output, names=["entity", "text_lower", "count", "doc_ids"])

        output_dir = os.path.join(os.path.dirname(__file__), "../output/statistics")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "entity_tags.csv")

        df.to_csv(path_or_buf=output_file, index=False)

        csv_dep_output = io.StringIO()
        csv_dep_writer = writer(csv_dep_output)

        for entity_tag in self.entity_statistics_dict:
            for dependency_tags_str in self.entity_statistics_dict[entity_tag].dependency_dict:
                doc_ids = self.entity_statistics_dict[entity_tag].dependency_dict[dependency_tags_str]
                csv_dep_writer.writerow([entity_tag, dependency_tags_str, len(doc_ids), doc_ids])

        csv_dep_output.seek(0)

        df = pd.read_csv(filepath_or_buffer=csv_dep_output, names=["entity", "dependency", "count", "doc_ids"])
        output_file = os.path.join(output_dir, "entity_dependency.csv")

        df.to_csv(path_or_buf=output_file, index=False)


class EntityStatistics:
    """Statistics of an entity type"""
    def __init__(self, entity_tag):
        self.count = 0
        self.entity_tag = entity_tag
        self.text_dict = dict()  # key: text,  value: list of doc ids
        self.dependency_dict = dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="store", default="C:/KA/lib/WNUT_2020/data/train_data/", dest="data_dir")

    args = parser.parse_args()

    ann_stats_obj = AnnotationStatistics()
    ann_stats_obj.process_dataset(data_dir=args.data_dir)
    ann_stats_obj.display_entity_tag_stats()
    ann_stats_obj.write_annotation_data()
