#!/usr/bin/env python3

"""
Approach
--------
    CRF

Execution command example
-------------------------
    python -u -m src.crf --ann_format standoff > ./output/results/crf_standoff.txt 2>&1
"""

import argparse
import glob
import joblib
import os
import pandas as pd
import seqeval.metrics as seqeval_metrics
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import sklearn.metrics as sklearn_metrics
import time
import traceback

from src.annotation import *
from src.dataset import *
from src.feature import *
from src.nlp_process import NLPProcess

class NER:
    def __init__(self):
        self.nlp_process_obj = None
        self.build_nlp_process()
        self.crf = None

    def build_nlp_process(self, verbose=True):
        self.nlp_process_obj = NLPProcess()
        self.nlp_process_obj.load_nlp_model(verbose=verbose)
        self.nlp_process_obj.build_sentencizer(verbose=verbose)

    def process_collection(self, data_dir, ann_format="conll"):
        X = []
        y = []

        for f in glob.glob(os.path.join(data_dir, "Standoff_Format", "*.txt")):
            # extract the protocol_id
            file_basename, _ = os.path.splitext(os.path.basename(f))
            protocol_id = file_basename[len("protocol_"):]

            # print("protocol_id: {}".format(protocol_id))

            file_document = os.path.join(data_dir, "Standoff_Format/protocol_" + protocol_id + ".txt")

            if ann_format == "conll":
                file_ann = os.path.join(data_dir, "Conll_Format/protocol_" + protocol_id + "_conll.txt")
            elif ann_format == "standoff":
                file_ann = os.path.join(data_dir, "Standoff_Format/protocol_" + protocol_id + ".ann")
            else:
                assert False, "Expected ann_format: a) conll,  b) standoff. Received: {}".format(ann_format)

            if not os.path.exists(file_document):
                print("{} not available".format(file_document))
                continue

            if not os.path.exists(file_ann):
                print("{} not available".format(file_ann))
                continue

            try:
                document_obj = Document(doc_id=int(protocol_id), nlp_process_obj=self.nlp_process_obj)
                document_obj.parse_document(document_file=file_document)
                if ann_format == "conll":
                    document_obj.parse_conll_annotation(conll_ann_file=file_ann)
                elif ann_format == "standoff":
                    document_obj.parse_standoff_annotation(ann_file=file_ann)

                feature_obj = Feature(doc_obj=document_obj)
                doc_features = feature_obj.extract_document_features(verbose=False)

                X.extend(doc_features)

                # populate document NER labels
                doc_named_entity_labels = []
                for sent in document_obj.sentences:
                    sent_named_entity_labels = [document_obj.words[word_index].named_entity_label
                                                for word_index in range(sent.start_word_index, sent.end_word_index)]
                    doc_named_entity_labels.append(sent_named_entity_labels)

                y.extend(doc_named_entity_labels)
            except:
                print("Failed for protocol_id: {}".format(protocol_id))
                traceback.print_exc()

        return X, y

    def train(self, X_train, y_train, ann_format="conll"):
        print("train begin")
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

        self.crf.fit(X=X_train, y=y_train)
        print("train end")

        model_dir = os.path.join(os.path.dirname(__file__), "../output/models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file_path = os.path.join(model_dir, "model_" + ann_format+".pkl")
        joblib.dump(value=self.crf, filename=model_file_path)
        print("Model saved: {}".format(model_file_path))

    def load_model(self, filename):
        self.crf = joblib.load(filename=filename)

    def evaluate(self, X_test, y_test, ann_format="conll"):
        """Evaluate

            Reference:
            ---------
                https://towardsdatascience.com/entity-level-evaluation-for-ner-task-c21fb3a8edf
                - seqeval for entity level metrics
        """
        y_pred = self.crf.predict(X=X_test)

        labels = list(self.crf.classes_)
        labels.remove('O')
        print("labels: {}".format(labels))

        print("\n---- Token level metrics ----\n")
        f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
        print("F1: {}".format(f1_score))

        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        sorted_labels.append('O')

        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))

        print("\n---- Entity level metrics ----\n")
        print("F1: {}".format(seqeval_metrics.f1_score(y_true=y_test, y_pred=y_pred)))

        print(seqeval_metrics.classification_report(y_true=y_test, y_pred=y_pred, digits=3))

        # confusion matrix
        # convert sequence of sequence into sequence
        y_test_arr = []
        for elem in y_test:
            y_test_arr.extend(elem)

        y_pred_arr = []
        for elem in y_pred:
            y_pred_arr.extend(elem)

        conf_matrix = sklearn_metrics.confusion_matrix(y_test_arr, y_pred_arr, labels=sorted_labels)
        df = pd.DataFrame(data=conf_matrix, index=sorted_labels, columns=sorted_labels, dtype=int)

        confusion_matrix_dir = os.path.join(os.path.dirname(__file__), "../output/confusion_matrix")

        if not os.path.exists(confusion_matrix_dir):
            os.makedirs(confusion_matrix_dir)

        confusion_matrix_file = os.path.join(confusion_matrix_dir, "confusion_matrix_" + ann_format + ".csv")
        df.to_csv(path_or_buf=confusion_matrix_file, index=True)

        # pd.set_option('display.max_rows', len(sorted_labels)+5)
        # pd.set_option('display.max_columns', len(sorted_labels))
        # print("\n------Confusion Matrix -----\n")
        # print(df)

    def predict_collection(self, input_data_dir, output_data_dir, ann_format="standoff"):
        """Predict Named Entity for each of the protocol in the input_data_dir
        """

        assert ann_format == "standoff", "As of now, only standoff format supported."

        # Iterate over each of the protocols
        for f in glob.glob(os.path.join(input_data_dir, "Standoff_Format", "*.txt")):
            # extract the protocol_id
            file_basename, _ = os.path.splitext(os.path.basename(f))
            protocol_id = file_basename[len("protocol_"):]

            try:
                self.predict_document(protocol_id=protocol_id, input_data_dir=input_data_dir, output_data_dir=output_data_dir, ann_format=ann_format)
            except:
                print("Failed for protocol_id: {}".format(protocol_id))
                traceback.print_exc()

    def predict_document(self, protocol_id, input_data_dir, output_data_dir, ann_format="standoff"):
        file_document = os.path.join(input_data_dir, "Standoff_Format/protocol_" + protocol_id + ".txt")
        file_ann = os.path.join(input_data_dir, "Standoff_Format/protocol_" + protocol_id + ".ann")

        if not os.path.exists(file_document):
            print("{} not available".format(file_document))
            return

        if not os.path.exists(file_ann):
            print("{} not available".format(file_ann))
            return

        document_obj = Document(doc_id=int(protocol_id), nlp_process_obj=self.nlp_process_obj)
        document_obj.parse_document(document_file=file_document)
        if ann_format == "conll":
            document_obj.parse_conll_annotation(conll_ann_file=file_ann)
        elif ann_format == "standoff":
            document_obj.parse_standoff_annotation(ann_file=file_ann)

        feature_obj = Feature(doc_obj=document_obj)
        doc_features = feature_obj.extract_document_features(verbose=False)

        assert self.crf is not None, "pre-requisite: train crf"
        y_pred_doc = self.crf.predict(X=doc_features)
        assert len(y_pred_doc) == len(document_obj.sentences), "Mismatch in len(y_pred_doc): {} :: len(sentences): {}".format(
            len(y_pred_doc), len(document_obj.sentences))

        entity_annotations = []

        # iterate over each of the sentences and build annotations(if entity predicted)
        for sent_i in range(len(document_obj.sentences)):
            start_word_index_sent = document_obj.sentences[sent_i].start_word_index
            n_words_sent = document_obj.sentences[sent_i].end_word_index - start_word_index_sent
            y_pred_sent = y_pred_doc[sent_i]
            assert len(y_pred_doc[sent_i]) == n_words_sent,\
                "Mismatch in word counts for sentence #{} :: n_words_sent: {} :: y_pred_sent: {}".format(
                    sent_i, n_words_sent, len(y_pred_sent))

            # iterate over the words
            for word_i in range(n_words_sent):
                ner_tag = y_pred_sent[word_i]
                if ner_tag == "O":
                    continue

                ner_tag_tokens = ner_tag.split("-")
                assert len(ner_tag_tokens) > 1, "NER tag not in BIO format :: Named Entity: {}".format(ner_tag)
                if ner_tag_tokens[0] == "B":
                    # start of next named entity
                    entity_type = ner_tag[2:]
                    entity_id = "T" + str(len(entity_annotations)+1)
                    start_word_index_ent = start_word_index_sent + word_i
                    end_word_index_ent = start_word_index_ent + 1
                    start_char_pos_ent = document_obj.words[start_word_index_ent].start_char_pos
                    end_char_pos_ent = document_obj.words[end_word_index_ent-1].end_char_pos

                    entity_ann = EntityAnnotation(entity_id=entity_id, start_char_pos=start_char_pos_ent,
                                                  end_char_pos=end_char_pos_ent, start_word_index=start_word_index_ent,
                                                  end_word_index=end_word_index_ent, entity_type=entity_type)
                    entity_annotations.append(entity_ann)
                elif ner_tag_tokens[0] == "I":
                    # continuation of prev named entity
                    # Set end_word_index of entity to next word. If entity continues to next word also, then we will update this field again.
                    end_word_index_ent = start_word_index_sent + word_i + 1
                    entity_annotations[-1].end_word_index = end_word_index_ent
                    entity_annotations[-1].end_char_pos = document_obj.words[end_word_index_ent-1].end_char_pos
                else:
                    assert False, "ERROR :: ner_tag expected either B-<Entity Type> or I-<Entity Type>"

        # Write predicted named entities in .ann file
        data_dir = os.path.join(output_data_dir, "Standoff_Format")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        file_ann = os.path.join(output_data_dir, "Standoff_Format/protocol_" + protocol_id + ".ann")

        with io.open(file=file_ann, mode="w", encoding="utf-8") as fd:
            for entity_ann in entity_annotations:
                entity_text = document_obj.text[entity_ann.start_char_pos: entity_ann.end_char_pos]
                fd.write("{}\t{} {} {}\t{}\n".format(entity_ann.id, entity_ann.type, entity_ann.start_char_pos, entity_ann.end_char_pos, entity_text))


def main(args):
    start_time = time.time()
    ner_obj = NER()
    print("\nNER object initialization took: {:.3f} seconds\n".format(time.time() - start_time))

    if args.train_model is None:
        start_time = time.time()
        train_X, train_y = ner_obj.process_collection(data_dir=args.train_data_dir, ann_format=args.ann_format)
        print('\nprocess_collection() for train data took {:.3f} seconds\n'.format(time.time() - start_time))
        start_time = time.time()
        ner_obj.train(X_train=train_X, y_train=train_y, ann_format=args.ann_format)
        print('\ntrain CRF took {:.3f} seconds\\n'.format(time.time() - start_time))
    else:
        ner_obj.load_model(filename=args.train_model)

    if args.evaluate_collection:
        start_time = time.time()
        dev_X, dev_y = ner_obj.process_collection(data_dir=args.dev_data_dir, ann_format=args.ann_format)
        print('\nprocess_collection() for dev data took {:.3f} seconds\n'.format(time.time() - start_time))

        start_time = time.time()
        ner_obj.evaluate(X_test=dev_X, y_test=dev_y, ann_format=args.ann_format)
        print('\nEvaluate took {:.3f} seconds\n'.format(time.time() - start_time))

    if args.predict_collection:
        start_time = time.time()
        output_data_dir = os.path.join(os.path.dirname(__file__), "../output/predict", os.path.basename(args.test_data_dir))
        ner_obj.predict_collection(input_data_dir=args.test_data_dir, output_data_dir=output_data_dir)
        print('\npredict_collection() for test data took {:.3f} seconds\n'.format(time.time() - start_time))

    if args.predict_protocol_id is not None:
        output_data_dir = os.path.join(os.path.dirname(__file__), "../output/predict", os.path.basename(os.path.dirname(args.test_data_dir)))
        ner_obj.predict_document(protocol_id=args.predict_protocol_id, input_data_dir=args.test_data_dir, output_data_dir=output_data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", action="store", default="C:/KA/lib/WNUT_2020/data/train_data/", dest="train_data_dir")
    parser.add_argument("--dev_data_dir", action="store", default="C:/KA/lib/WNUT_2020/data/dev_data/",
                        dest="dev_data_dir")
    parser.add_argument("--test_data_dir", action="store", default="C:/KA/lib/WNUT_2020/data/test_data_2020", dest="test_data_dir")
    parser.add_argument("--ann_format", action="store", default="conll", dest="ann_format",
                        help="Either conll or standoff")
    parser.add_argument("--train_model", action="store", default=None, dest="train_model")
    parser.add_argument("--evaluate_collection", action="store_true", default=False, dest="evaluate_collection",
                        help="Evaluate NER on dev_data_dir protocols")
    parser.add_argument("--predict_collection", action="store_true", default=False, dest="predict_collection",
                        help="Predict NER on test_data_dir protocols")
    parser.add_argument("--predict_protocol_id", action="store", default=None, dest="predict_protocol_id",
                        help="Predict NER on a specific protocol of test_data_dir")


    args = parser.parse_args()
    main(args=args)
