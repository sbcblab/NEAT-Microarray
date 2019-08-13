#Bruno Iochins Grisci
#DECEMBER/2017

import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
    return filename+'.gct'     

if __name__ == '__main__':
    dataset_file = sys.argv[1]
    label_file = sys.argv[2]
    mrmr_file = sys.argv[3]
    
    #save_dir = 'mRMR_out/'
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    
    ds = MAR(dataset_file, label_file)
    dataset = ds.get_dataset(sampleXfeature=True, standardized=True)
    
    order = []
    fea   = []
    name  = [] 
    score = []
    
    with open(mrmr_file, 'r') as mf:
        while '*** mRMR features ***' not in mf.readline():
            pass
        line = mf.readline()
        print(line)
        while line.strip():
            line = mf.readline()
            if line.strip():
                l = line.split()
                print(l)
                order.append(int(l[0]))
                fea.append(int(l[1])-1)
                name.append(l[2])
                score.append(float(l[3]))        
    
    print(zip(order, fea, name, score))
    
    save_gene_selection(dataset_file, fea, mrmr_file.replace('.txt', ''))
    plot_genes_correlation(dataset, fea, ds.dataset_symbols, mrmr_file.replace('.txt', '_corr'))
    plot_selected_genes_expression(dataset, fea, ds.dataset_symbols, ds.get_classes_one_hot(), ds.classes_labels, mrmr_file.replace('.txt', '_exp'))
   
