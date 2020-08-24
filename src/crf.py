#!/usr/bin/env python

"""
Approach
--------
    CRF

Execution command
-----------------
    python -m src.crf --ann_format standoff
"""

import argparse
import glob
import os
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import traceback

from .dataset import *
from .feature import *
from .nlp_process import NLPProcess

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

            print("protocol_id: {}".format(protocol_id))

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

    def train(self, X_train, y_train):
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

    def evaluate(self, X_test, y_test):
        y_pred = self.crf.predict(X=X_test)

        labels = list(self.crf.classes_)
        labels.remove('O')
        print("labels: {}".format(labels))

        f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
        print("F1: {}".format(f1_score))

        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        ))

def main(args):
    ner_obj = NER()

    train_X, train_y = ner_obj.process_collection(data_dir=args.train_data_dir, ann_format=args.ann_format)
    ner_obj.train(X_train=train_X, y_train=train_y)

    dev_X, dev_y = ner_obj.process_collection(data_dir=args.dev_data_dir, ann_format=args.ann_format)
    ner_obj.evaluate(X_test=dev_X, y_test=dev_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", action="store", default="C:/KA/lib/WNUT_2020/data/train_data/", dest="train_data_dir")
    parser.add_argument("--dev_data_dir", action="store", default="C:/KA/lib/WNUT_2020/data/dev_data/",
                        dest="dev_data_dir")
    parser.add_argument("--ann_format", action="store", default="conll", dest="ann_format",
                        help="Either conll or standoff")

    args = parser.parse_args()
    main(args=args)
