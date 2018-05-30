#BRUNO IOCHINS GRISCI

from __future__ import print_function
import os
import sys
from shutil import copyfile
import math
import numpy as np
from collections import OrderedDict
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import ntpath

from microarray_reader import MAR

COLORS =  ['red', 'blue', 'green', 'k', 'yellow', 'pink', 'gray']
MARKERS = ['o',   's',    'v',     'x', 'D',      '+',    'h']

def convert_to_class(entry):
    if type(entry) is not list:
        entry = entry.tolist()
    return entry.index(max(entry))

def make_fig(data, colors, markers, labels, n_dim, n_samples, method, file_name, variance=None):
    
    file_name = ntpath.basename(file_name)
    
    fig = plt.figure()
    if n_dim == '3d':
        ax = fig.add_subplot(111, projection=n_dim)
    else:
        ax = fig.add_subplot(111)
    for i in xrange(n_samples):
        if n_dim == '2d':
            ax.scatter(data[i, 0], data[i, 1], c=colors[i], marker=markers[i], label=labels[i])
        elif n_dim == '3d':
            ax.scatter(data[i, 0], data[i, 1], data[i, 2], c=colors[i], marker=markers[i], label=labels[i])      
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    
    if variance is None:
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    else:
        plt.xlabel('Component 1 ({:.3f})'.format(variance[0]))
        plt.ylabel('Component 2 ({:.3f})'.format(variance[1]))
    plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.09), fancybox=True, ncol=4)
    plt.title(file_name.replace('.gct', method+n_dim))
    plt.savefig(save_dir+file_name.replace('.gct', method+n_dim+'.svg'), bbox_inches='tight')
    plt.close()     

if __name__ == '__main__':
    tr_data_file = sys.argv[1]
    tr_label_file = sys.argv[2]

    save_dir = tr_data_file.replace(os.path.basename(tr_data_file), '')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile('tSNE.py', save_dir+'t.py')

    #READ TRAINING DATASET
    training = MAR(tr_data_file, tr_label_file)

    number_samples = training.number_data_samples
    number_features = training.number_data_features
    number_classes = training.number_classes   
    classes_labels = training.classes_labels   
    print(number_samples, number_features, number_classes)
    print(classes_labels)

    tr_inputs = training.get_dataset(sampleXfeature=True, standardized=False)
    tr_min = tr_inputs.min(0)
    tr_ptp = tr_inputs.ptp(0)    
    tr_inputs = 2.0 * ((tr_inputs - tr_min)/tr_ptp) - 1.0    
    tr_outputs = training.get_classes_one_hot()

    classes = [classes_labels[convert_to_class(c)] for c in tr_outputs]
    colors  = [COLORS[convert_to_class(c)]  for c in tr_outputs]
    markers = [MARKERS[convert_to_class(c)] for c in tr_outputs]
    print(classes)
    print(colors)

    print(tr_inputs.shape)
    
    pca2D = PCA(n_components=2)
    pca_2 = pca2D.fit_transform(tr_inputs)
    print(pca_2.shape)
    pca2D_var = pca2D.explained_variance_ratio_
    print(pca2D_var)
    #pca_3 = PCA(n_components=3).fit_transform(tr_inputs)
    #print(pca_3.shape)    
  
    tSNE2D = TSNE(n_components=2)
    tSNE_2 = tSNE2D.fit_transform(tr_inputs)
    print(tSNE_2.shape)
    #tSNE_3 = TSNE(n_components=3).fit_transform(tr_inputs)
    #print(tSNE_3.shape)   
  
    if number_features > 50 and number_samples > 50:  
        pca50D = PCA(n_components=50)
        pca_50 = pca50D.fit_transform(tr_inputs)
        print(pca_50.shape)  
        pca50D_var = pca50D.explained_variance_ratio_
        print(pca50D_var)  
        emb2D = TSNE(n_components=2)   
        embedded_2 = emb2D.fit_transform(pca_50)
        print(embedded_2.shape)
        #embedded_3D = TSNE(n_components=3).fit_transform(pca_50)
        #print(embedded_3D.shape)   
        make_fig(embedded_2, colors, markers, classes, '2d', number_samples, 'PCA+tSNE', tr_data_file, variance=None)
        #make_fig(embedded_3D, colors, markers, classes, '3d', number_samples, 'PCA+tSNE', tr_data_file)      
    
    make_fig(pca_2, colors, markers, classes, '2d', number_samples, 'PCA', tr_data_file, variance=pca2D_var)
    #make_fig(pca_3, colors, markers, classes, '3d', number_samples, 'PCA', tr_data_file) 

    make_fig(tSNE_2, colors, markers, classes, '2d', number_samples, 'tSNE', tr_data_file, variance=None)
    #make_fig(tSNE_3, colors, markers, classes, '3d', number_samples, 'tSNE', tr_data_file)  
