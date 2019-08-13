#BRUNO IOCHINS GRISCI

from __future__ import print_function
import os
import sys
from shutil import copyfile
import math
import numpy as np
from scipy import stats
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import neatB.math_util

import log_error
import ML_frame
import mrmr2gct
from microarray_reader import MAR

def convert_to_class(entry):
    if type(entry) is not list:
        entry = entry.tolist()
    return entry.index(max(entry))

def kruskal_wallis(inputs, outputs, features_names, save_dir):
    print('Computing Kruskal Wallis Test for {} genes with {} samples in {} classes!'.format(inputs.shape[1], inputs.shape[0], outputs.shape[1]))
    genes_pvalues = []
    for gene in xrange(inputs.shape[1]):
        gene_expression = inputs[:,gene]
        classes = []
        for o in xrange(outputs.shape[1]):
            classes.append([])
        for sample in zip(outputs, gene_expression):
             classes[convert_to_class(sample[0])].append(sample[1])
        genes_pvalues.append(stats.kruskal(*classes)[1])

    with open(save_dir+'kwt.txt', 'w') as kwt_file:
        for gene in xrange(len(genes_pvalues)):
            kwt_file.write('{:20s}: {}\n'.format(features_names[gene], genes_pvalues[gene]))       

    with open(save_dir+'kwt_sort.txt', 'w') as kwts_file:
        sor = zip(features_names, genes_pvalues)
        sor.sort(key=lambda tup: tup[1])
        for v in sor:
            kwts_file.write('{:20s}: {}\n'.format(v[0], v[1])) 
        
    return np.array(genes_pvalues)

def get_probs(pvalues, features_names, save_dir):
    c = -1.0
    b = 10.0
    probs = neatB.math_util.softmax(c * (np.log(pvalues)/np.log(b)))
    print('Computing probabilities for {} genes with {} log {}'.format(len(pvalues), c, b))
    with open(save_dir+'probs.txt', 'w') as probs_file:
        for gene in xrange(len(pvalues)):
            probs_file.write('{:20s}: {}\n'.format(features_names[gene], probs[gene]))
    
    with open(save_dir+'probs_sort.txt', 'w') as probss_file:
        sor = zip(features_names, probs)
        sor.sort(key=lambda tup: tup[1], reverse=True) 
        for v in sor:
            probss_file.write('{:20s}: {}\n'.format(v[0], v[1]))

    sor = np.sort(probs)[::-1]
    try:
        print(10, np.sum(sor[:10]))
        print(100, np.sum(sor[:100])) 
        print(1000, np.sum(sor[:1000])) 
        print(10000, np.sum(sor[:10000]))  
    except:
        pass 
                      
    return probs

def kw_select(pvalues, features_names, save_dir, tr_inputs, te_inputs=None, threshold = 0.01):
    global tr_data_file
    gct_file = save_dir+'kw'
    sel_p   = []
    sel_tri = []
    sel_tei = None
    sel_fn  = []
    fea     = []
    
    for i in xrange(len(pvalues)):
        if pvalues[i] <= threshold:
            sel_p.append(pvalues[i])
            sel_fn.append(features_names[i])
            fea.append(i)

    not_selected = np.delete(np.arange(len(features_names)), np.array(sorted(fea)))
    sel_tri = np.delete(tr_inputs, not_selected, 1)
    if te_inputs is not None:   
        sel_tei = np.delete(te_inputs, not_selected, 1)

    mrmr2gct.save_gene_selection(tr_data_file, fea, gct_file)
    print('Selected {} genes with p < {} from a total of {} genes and {} samples!'.format(sel_tri.shape[1], threshold, tr_inputs.shape[1], sel_tri.shape[0]))
    return sel_p, sel_tri, sel_tei, sel_fn, gct_file + '.gct'
       
def run(clf, tr_inputs, tr_outputs, te_inputs=None, te_outputs=None, tr_names=None, te_names=None, features_names=None, classes_labels=None, save_dir='', num_cores=1, num_gen=1):
    global use_kw
    outputs = [ML_frame.convert_to_class(x) for x in tr_outputs]
    
    ############################################################################
    if use_kw > 0:
        pvalues = kruskal_wallis(tr_inputs, tr_outputs, features_names, save_dir)
        print(tr_inputs.shape)
        if te_inputs is not None:
            print(te_inputs.shape)
        sel_gct_file = tr_data_file
        pvalues, tr_inputs, te_inputs, features_names, sel_gct_file = kw_select(pvalues, features_names, save_dir, tr_inputs, te_inputs)
        print(tr_inputs.shape)
        if te_inputs is not None:
            print(te_inputs.shape)
        probs = get_probs(pvalues, features_names, save_dir)

    ############################################################################

    print('Scaling the training and the testing sets!')
    
    tr_mean = np.mean(tr_inputs, axis=0)
    tr_ptp = tr_inputs.ptp(0)       
    tr_inputs = (tr_inputs - tr_mean)/tr_ptp
    if te_inputs is not None:
        te_inputs = (te_inputs - tr_mean)/tr_ptp
    print(tr_inputs.shape)
    if te_inputs is not None:
        print(te_inputs.shape)    
    
    ############################################################################    
    
    #CREATE CLASSIFIER
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    degree_range = np.array([3,4])
    tol_range = np.array([0.001, 0.01])
    param_grid = dict(gamma=gamma_range, C=C_range, degree=degree_range, tol=tol_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(tr_inputs, outputs)
    #print("The best parameters are %s with a score of %0.2f"
    #      % (grid.best_params_, grid.best_score_))        
    '''clf = SVC(C=10.0, cache_size=1000, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)'''
    clf = grid.best_estimator_

    clf.fit(tr_inputs, outputs)
    
    ERROR_NAMES = []
    bla, bla2, error_names = log_error.report(tr_inputs, tr_outputs, classes_labels, names=tr_names, classifier=clf, error_file_path=save_dir+'training_error.txt', conf_matrix=True, dataset_label='TRAINING')
    for en in error_names:
        ERROR_NAMES.append(en)    
    k_conf_matrix = None
    acc = 0.0
    if te_inputs is not None and te_outputs is not None:
        k_conf_matrix, acc, error_names = log_error.report(te_inputs, te_outputs, classes_labels, names=te_names, classifier=clf, error_file_path=save_dir+'testing_error.txt', conf_matrix=True, dataset_label='TESTING')    
        for en in error_names:
            ERROR_NAMES.append(en)    
    return k_conf_matrix, acc, ERROR_NAMES, 0    

if __name__ == '__main__':
    save_dir  = sys.argv[1] + '/'
    use_kw = int(sys.argv[2])
    k = int(sys.argv[3])
    tr_data_file = sys.argv[4]
    tr_label_file = sys.argv[5]
    indexes_file = sys.argv[6]
    try:
        te_data_file = sys.argv[7]
        te_label_file = sys.argv[8]
        independent_test = True
        print('Independent test set used')
    except IndexError:
        independent_test = False

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile('SVM.py', save_dir+'s.py')

    #READ TRAINING DATASET
    training = MAR(tr_data_file, tr_label_file)

    #READ PRECOMPUTED CROSS-VALIDATION INDEXES
    if indexes_file == 'None':
        pre_indexes = None
    else:
        pre_indexes = ML_frame.get_pre_folds(k, indexes_file)   

    #CREATE CLASSIFIER
    #C_range = np.logspace(-2, 10, 13)
    #gamma_range = np.logspace(-9, 3, 13)
    #param_grid = dict(gamma=gamma_range, C=C_range)
    #cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    #grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    #grid.fit(tr_inputs, outputs)
    #print("The best parameters are %s with a score of %0.2f"
    #      % (grid.best_params_, grid.best_score_))        
    #clf = SVC(C=10.0, cache_size=1000, class_weight=None, coef0=0.0,
    #            decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
    #            max_iter=-1, probability=False, random_state=None, shrinking=True,
    #            tol=0.001, verbose=False)
    #clf = SVC(grid.best_params_)
    clf = None

    if independent_test:
        testing = MAR(te_data_file, te_label_file)
        ML_frame.start(clf, run, training, testing=testing, k=k, save_dir=save_dir, pre_indexes=pre_indexes)
    else:    
        ML_frame.start(clf, run, training, k=k, save_dir=save_dir, pre_indexes=pre_indexes)
