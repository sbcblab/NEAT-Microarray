#BRUNO IOCHINS GRISCI

from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from shutil import copyfile
import math
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import seaborn as sns

import visualize
import log_error
import ML_frame
from microarray_reader import MAR

def plot_selected_genes_expression(dataset, indexes, genes_names, classes, classes_labels, filename):
    names = [ genes_names[i] for i in indexes]
    #indexes, names = zip(*sorted(zip(indexes, names)))     
    classes = [classes_labels[ML_frame.convert_to_class(c)] for c in classes]
    data = dataset[:, indexes]
    
    sns.set(style="white")
    I = pd.Index(classes, name="Samples")
    C = pd.Index(names, name="Genes")
    d = pd.DataFrame(data=data, index=I, columns=C).transpose()
    
    print(dataset.shape)
    print(data.shape)
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(40,20))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(h_neg=190, h_pos=100, l=50, center='dark', as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    hm = sns.heatmap(d, cmap=cmap, vmin=-2.0, vmax=2.0, center=0.0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    fig = hm.get_figure()
    fig.savefig(filename+'.svg')  
    
    with open(filename+'.txt', 'w') as df:
        df.write(d.to_csv())                  

def save_gene_selection(data_file, genes, filename):
    indexes = [i+3 for i in genes] + [0, 1, 2]
    with open(filename+'.gct', 'w') as nf:
        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                if i in indexes:
                    if i == 1:
                        number_genes = len(genes)
                        info_line = map(int, line.split())
                        number_data_samples = info_line[1]
                        line = '{:d} {:d}\n'.format(number_genes, number_data_samples)
                    nf.write(line)

def plot_genes_correlation(dataset, indexes, genes_names, filename):
    names = [ genes_names[i] for i in indexes]
    #indexes, names = zip(*sorted(zip(indexes, names)))      
    data = dataset[:, indexes]
    
    sns.set(style="white")
    d = pd.DataFrame(data=data, columns=names)
    # Compute the correlation matrix
    corr = d.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    hm = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    fig = hm.get_figure()
    fig.savefig(filename+'.svg')            

def plot_genes_perturbation(dataset, indexes, genes_names, classes_labels, filename, color='Navy'):    
    sns.set(style="white")
    data = dataset[indexes,:]
    genes = [ genes_names[i] for i in indexes]
    I = pd.Index(genes, name="Genes")
    C = pd.Index(['Total'] + classes_labels, name="Classes")
    d = pd.DataFrame(data=data, index=I, columns=C).transpose()
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(200,30))
    ax=plt.subplot(111)
    # Draw the heatmap with the mask and correct aspect ratio
    #cmap = sns.palplot(sns.color_palette("Reds", as_cmap=True))
    cmap = sns.light_palette(color, as_cmap=True)
    hm = sns.heatmap(d, cmap=cmap, vmin=0.0, vmax=1.0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    fig = hm.get_figure()
    fig.savefig(filename+'.svg')  
    
    with open(filename+'.txt', 'w') as df:
        df.write(d.to_csv()) 

def accuracy(clf, inputs, outputs):
    error = 0.0
    prediction = []
    for i, o in zip(inputs, outputs):
        if type(o) is not list:
            o = o.tolist()    
        output = clf.predict([i]).tolist()[0]  
        prediction.append(output)  
        if max(output) == min(output) or o.index(max(o)) != output.index(max(output)):
            error += 1.0
    error = error / len(inputs)
    
    prediction = np.array(prediction)
    outputs = np.array(outputs)
    
    misclassification = outputs - prediction
    correct_label = []
    incorrect_label = []
    true_label = []
    fake_label = [] 
    for label in xrange(outputs.shape[1]):  
        correct_label.append(len(np.where(outputs[:,label] == 1)[0]))
        incorrect_label.append(len(np.where(misclassification[:,label] == 1)[0]))
        true_label.append(len(np.where(outputs[:,label] == 0)[0]))
        fake_label.append(len(np.where(misclassification[:,label] == -1)[0]))
    
    error_by_class = [float(x)/float(y) for x, y in zip(incorrect_label, correct_label)]
    fakes = [float(x)/float(y) for x, y in zip(fake_label, true_label)]
    #print(error_by_class)
    return error, error_by_class, fakes

def c9(clf, inputs, outputs):
    maxs = []
    mins = []
    c9 = 0.0
    for gene in xrange(inputs.shape[1]):
        gin = np.ones((inputs.shape[1]))
        gin[gene] = 0.0    
        output = clf.predict_proba([gin]).tolist()[0]
        maxs.append(max(output))
        mins.append(min(output))
        if max(output) > 0.9:
            c9 += 1
        #print(output)
    print('c9', c9)
    print('max mean: ', np.mean(np.array(maxs)), 'std: ', np.std(np.array(maxs)), 'max: ', np.amax(np.array(maxs)), 'min: ', np.amin(np.array(maxs)))  
    print('min mean: ', np.mean(np.array(mins)), 'std: ', np.std(np.array(mins)), 'max: ', np.amax(np.array(mins)), 'min: ', np.amin(np.array(mins)))       

def occlude(clf, inputs, outputs, genes_names, classes_labels, filename, te_inputs=None, te_outputs=None):
    
    nog = 1
    genes_error = []
    indexes = []
    occlusion = np.full((inputs.shape[0], 1), 40.0)
    print(occlusion)
    print(occlusion.shape)
    data = []
    dota = []
    
    for gene in xrange(inputs.shape[1]):
        transformed = np.copy(inputs)
        transformed[:,gene:gene+nog] = occlusion
        a, ebc, f = accuracy(clf, transformed, outputs)
        #print(gene, a, ebc, f)
        line = [a] + ebc
        lone = [a] + f
        data.append(line)
        dota.append(lone)
        if max(line) > 0.0 or max(lone) > 0.0:
            indexes.append(gene)
        genes_error.append(a)
        #print(gene, gene+nog, a)
         
    data = np.array(data)
    print('DDDDDDDDDD')
    print(data)
    print(data.shape)
    print('IMPORTANT:')    
    print(len(indexes))
    plot_genes_perturbation(data, indexes, genes_names, classes_labels, filename+'FN')    
    dota = np.array(dota)
    print('ddddddddd')
    print(dota)
    print(dota.shape)   
    plot_genes_perturbation(dota, indexes, genes_names, classes_labels, filename+'FP', color='green')     
    save_gene_selection(tr_data_file, indexes, filename)    
    plot_selected_genes_expression(inputs, indexes, genes_names, outputs, classes_labels, filename+'_exp')
    plot_genes_correlation(inputs, indexes, genes_names, filename+'_corr')
    
    if te_inputs is not None and te_outputs is not None:
        #plot_genes_perturbation(data, indexes, genes_names, classes_labels, filename)        
        plot_selected_genes_expression(te_inputs, indexes, genes_names, te_outputs, classes_labels, filename+'_exp_TEST')
        plot_genes_correlation(te_inputs, indexes, genes_names, filename+'_corr_TEST')
    
    genes_error = np.array(genes_error)
    print('------------------------')
    print(np.sort(genes_error))
    print(genes_error.shape)
    print('mean', np.mean(genes_error))
    print('std',  np.std(genes_error))
    print('max', np.amax(genes_error))
    print('i_max', np.where(genes_error > np.mean(genes_error) + np.std(genes_error)))
    print('MAX', np.argmax(genes_error))
    print('min', np.amin(genes_error)) 
    print('i_min', np.where(genes_error < np.mean(genes_error) - np.std(genes_error)))
    print('MIN', np.argmin(genes_error))   
    
    print(inputs.shape)
    print(np.amax(inputs), np.amin(inputs))
    print(np.array(outputs).shape)
    print(occlusion.shape)
    print(data.shape)
    print(len(genes_names))
               
def run(clf, tr_inputs, tr_outputs, te_inputs=None, te_outputs=None, features_names=None, classes_labels=None, save_dir='', num_cores=1, num_gen=1):
    clf.fit(tr_inputs, tr_outputs)
    print([coef.shape for coef in clf.coefs_])
    for coef in clf.coefs_:
        print(coef)
        print('MEAN: {} - STD: {} - MAX: {} - MIN {}'.format(np.mean(coef), np.std(coef), np.amax(coef), np.amin(coef)))
    
    first_layer = clf.coefs_[0]
    #print([first_layer > np.mean(first_layer) + np.std(first_layer)])
    '''indexes = np.where(first_layer > np.mean(first_layer) + 10.0*np.std(first_layer))
    lin = indexes[0]
    col = indexes[1]
    genes = set()
    for l in lin:
        genes.add(features_names[l])
    print(len(genes)) 
    print(genes)'''
    maxdex = np.where(first_layer == np.amax(first_layer))
    print(maxdex)
    for i in maxdex[0]:
        print(features_names[i])
    mindex = np.where(first_layer == np.amin(first_layer))
    print(mindex)
    for i in mindex[0]:
        print(features_names[i]) 
    
    for i in xrange(len(clf.coefs_)):
        visualize.plot_matrix(clf.coefs_[i], save_dir+'layer{}'.format(i+1),5,50)             

    with open(save_dir+'coefs.txt', 'w') as coefs_file:
        coefs_file.write('{!s}'.format([coef.shape for coef in clf.coefs_]))
        coefs_file.write('\n{!s}'.format(clf.coefs_))

    log_error.report(tr_inputs, tr_outputs, classes_labels, classifier=clf, error_file_path=save_dir+'training_error.txt', conf_matrix=True, dataset_label='TRAINING')
    k_conf_matrix = None
    acc = 0.0
    if te_inputs is not None and te_outputs is not None: 
        k_conf_matrix, acc = log_error.report(te_inputs, te_outputs, classes_labels, classifier=clf, error_file_path=save_dir+'testing_error.txt', conf_matrix=True, dataset_label='TESTING')    
    
    
    #occlude(clf, tr_inputs, tr_outputs, features_names, classes_labels, save_dir+'disturbance', te_inputs, te_outputs)
    #accuracy(clf, tr_inputs, tr_outputs)
    #c9(clf, tr_inputs, tr_outputs)
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

    save_dir = 'mlp_out/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile('mlp.py', save_dir+'m.py')

    #READ TRAINING DATASET
    training = MAR(tr_data_file, tr_label_file)

    #CREATE CLASSIFIER
    clf = MLPClassifier(activation='tanh', alpha=1.0, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=(5, ), learning_rate='constant',
           learning_rate_init=0.001, max_iter=900, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.2, verbose=False,
           warm_start=False)

    if independent_test:
        testing = MAR(te_data_file, te_label_file)
        ML_frame.start(clf, run, training, testing=testing, k=k, save_dir=save_dir)
    else:    
        ML_frame.start(clf, run, training, k=k, save_dir=save_dir)
