#BRUNO IOCHINS GRISCI

from __future__ import print_function
import os
import sys
from shutil import copyfile
import math
from sklearn import tree
import numpy as np
import graphviz

import log_error
import ML_frame
from microarray_reader import MAR

def run(clf, tr_inputs, tr_outputs, te_inputs=None, te_outputs=None, features_names=None, classes_labels=None, save_dir='', num_cores=1, num_gen=1):
    outputs = [ML_frame.convert_to_class(x) for x in tr_outputs]
    clf.fit(tr_inputs, outputs)

    log_error.report(tr_inputs, tr_outputs, classes_labels, classifier=clf, error_file_path=save_dir+'training_error.txt', conf_matrix=True, dataset_label='TRAINING')
    k_conf_matrix = None
    acc = 0.0
    if te_inputs is not None and te_outputs is not None:
        k_conf_matrix, acc = log_error.report(te_inputs, te_outputs, classes_labels, classifier=clf, error_file_path=save_dir+'testing_error.txt', conf_matrix=True, dataset_label='TESTING')    
    
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=features_names,  
                                    class_names=classes_labels,  
                                    filled=True, rounded=True,  
                                    special_characters=True)   
    graph = graphviz.Source(dot_data) 
    graph.render(save_dir+"dtree")    
    
    return k_conf_matrix, acc     
   
if __name__ == '__main__':   
    k = int(sys.argv[1])
    tr_data_file = sys.argv[2]
    tr_label_file = sys.argv[3]
    try:
        te_data_file = sys.argv[4]
        te_label_file = sys.argv[5]
        independent_test = True
        print('Independent test set used')
    except IndexError:
        independent_test = False

    save_dir = 'dt_outLOOCV/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile('decision_tree.py', save_dir+'dt.py')

    #READ TRAINING DATASET
    training = MAR(tr_data_file, tr_label_file)

    #CREATE CLASSIFIER
    clf = tree.DecisionTreeClassifier()

    if independent_test:
        testing = MAR(te_data_file, te_label_file)
        ML_frame.start(clf, run, training, testing=testing, k=k, save_dir=save_dir)
    else:    
        ML_frame.start(clf, run, training, k=k, save_dir=save_dir)
