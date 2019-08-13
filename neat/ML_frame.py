import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter

#BRUNO IOCHINS GRISCI

import con_matrix
import visualize
from microarray_reader import MAR

def convert_to_class(entry):
    if type(entry) is not list:
        entry = entry.tolist()
    return entry.index(max(entry))

def get_pre_folds(k, indexes_file):
    pre_indexes = []
    with open(indexes_file, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content] 
        content = [x.replace('TR,', '') for x in content]
        content = [x.replace('TE,', '') for x in content]
    for tr, te in zip(content[0::2], content[1::2]):
        tri = np.fromstring(tr, dtype=int, sep=',')
        tei = np.fromstring(te, dtype=int, sep=',')
        pre_indexes.append([tri, tei])
    if len(pre_indexes) != k:
        raise ValueError('The number of pre computed folds does not match the informed k.')
    return pre_indexes

def start(clf, f, training, testing=None, k=0, save_dir='', num_cores=1, num_gen=1, pre_indexes=None):
    #READ TRAINING DATASET    
    number_samples = training.number_data_samples
    number_features = training.number_data_features
    number_classes = training.number_classes   
    features_labels = training.dataset_symbols
    classes_labels = training.classes_labels
    print(number_samples, number_features, number_classes)
    print(classes_labels)

    #K-FOLD CROSS-VALIDATION
    k_conf_matrix = None
    ACC = []
    FS  = []
    ERROR_NAMES = []
    acc = 0.0
    if k > 0:
        trx_folds, try_folds, tex_folds, tey_folds, tr_names, te_names = training.get_k_folds(k, standardized=False, save_dir=save_dir, pre_indexes=pre_indexes)
        i = 1
        for trx, tryy, tex, tey, tr_n, te_n in zip(trx_folds, try_folds, tex_folds, tey_folds, tr_names, te_names):
            print(str(i) + '-FOLD ###############################################')
            sd = save_dir + str(i) +'fold/'
            if not os.path.exists(sd):
                os.makedirs(sd)       
            kcm, acci, enames, nfeatures = f(clf, trx, tryy, tex, tey, tr_names=tr_n, te_names=te_n, features_names=features_labels, classes_labels=classes_labels, save_dir=sd, num_cores=num_cores, num_gen=num_gen)
            if k_conf_matrix is None:
                k_conf_matrix = kcm
            else:
                k_conf_matrix += kcm
            acc += (float(len(tey)) / float(number_samples)) * acci
            ACC.append(acci)
            FS.append(nfeatures)
            for en in enames:
                ERROR_NAMES.append(en)
            with open(save_dir+'accuracy.txt', 'a') as acc_file:
                acc_file.write('{:2d}-FOLD accuracy: {:3f}\n'.format(i, acci))
            with open(save_dir+'feature_selection.txt', 'a') as fs_file:
                fs_file.write('{:2d}-FOLD feature selection: {:3f}\n'.format(i, nfeatures))                 
            #visualize.plot_matrix(trx, sd+'train_inputs', 1,1, tryy) 
            #visualize.plot_matrix(tex, sd+'test_inputs', 1,1, tey)        
            i += 1      
    
        #REPORT RESULTS FROM K-FOLD CROSS-VALIDATION
        with open(save_dir+'accuracy.txt', 'a') as acc_file:
            acc_file.write('{:2d}-FOLD CROSS VALIDATION average accuracy: {:3f}\n---\n'.format(k, acc))
            acc_file.write('Average accuracy: {}\n'.format(np.array(ACC).mean()))
            acc_file.write('Std accuracy:     {}'.format(np.array(ACC).std())) 
        print(str(k) + '-FOLD CROSS-VALIDATION RESULTS #############')        
        print('Average accuracy: ' + str(acc))
        plt.figure()
        con_matrix.plot_confusion_matrix(k_conf_matrix, classes=classes_labels, file_name=save_dir+'cv_cm.svg')
        plt.figure()
        con_matrix.plot_confusion_matrix(k_conf_matrix, classes=classes_labels, file_name=save_dir+'cv_ncm.svg', normalize=True, title='Normalized confusion matrix')
 
        with open(save_dir+'feature_selection.txt', 'a') as fs_file:
            fs_file.write('{:2d}-FOLD CROSS VALIDATION average feature selection: {:3f}\n---\n'.format(k, np.array(FS).mean()))
            fs_file.write('Average feature selection: {}\n'.format(np.array(FS).mean()))
            fs_file.write('Std feature selection:     {}'.format(np.array(FS).std()))         
        print('Average feature selection: ' + str(np.array(FS).mean())) 
        
    #FINAL RUN WITH ALL TRAINING AND OPTIONAL INDEPENDENT TEST SET
    print('FINAL ###############################################')
    tr_mean, tr_std = training.get_mean_std()
    print('TRAIN mean, std: ', tr_mean, tr_std)
    tr_inputs = training.get_dataset(sampleXfeature=True, standardized=False)
    #tr_inputs = training.get_dataset(sampleXfeature=True, rescaled=True, min_value=0.0, max_value=1.0, standardized=False, mean=None, std=None)
    tr_outputs = np.array(training.get_classes_one_hot())
    
    #visualize.plot_matrix(tr_inputs, save_dir+'train_inputs', 1,1, tr_outputs) 
    if testing is not None:
        print(testing.number_data_samples, testing.number_data_features, testing.number_classes)
        print('TEST mean, std: ', testing.get_mean_std())
        te_inputs = testing.get_dataset(sampleXfeature=True, standardized=False, mean=tr_mean, std=tr_std)
        te_outputs = np.array(testing.get_classes_one_hot()) 
        kcm, acc, enames, nfeatures = f(clf, tr_inputs, tr_outputs, te_inputs, te_outputs, tr_names=training.dataset_header[2:], te_names=testing.dataset_header, features_names=features_labels, classes_labels=classes_labels, save_dir=save_dir, num_cores=num_cores, num_gen=num_gen)
        for en in enames:
            ERROR_NAMES.append(en)        
        with open(save_dir+'accuracy.txt', 'a') as acc_file:
            acc_file.write('INDEPENDENT TEST SET accuracy: {:3f}\n'.format(acc)) 
        visualize.plot_matrix(te_inputs, save_dir+'test_inputs', 1,1, te_outputs) 
        with open(save_dir+'feature_selection.txt', 'a') as fs_file:
            fs_file.write('INDEPENDENT TEST SET feature_selection: {:3f}\n'.format(nfeatures))             
    '''else:
        bla, bla2, enames, nfeatures = f(clf, tr_inputs, tr_outputs, tr_names=training.dataset_header[2:], te_names=None, features_names=features_labels, classes_labels=classes_labels, save_dir=save_dir, num_cores=num_cores, num_gen=num_gen)
        for en in enames:
            ERROR_NAMES.append(en)'''
            
    counter = Counter(ERROR_NAMES)
    with open(save_dir+'error_names.txt', 'w') as name_file:
        s = sorted(counter, key=lambda x: counter[x], reverse=True)
        for k in s:
            name_file.write('[{:2d}] {:15s}: {}\n'.format(int(counter[k]), k, int(counter[k]) * '*'))                    
