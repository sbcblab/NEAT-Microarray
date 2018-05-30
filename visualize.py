from __future__ import print_function

import copy
import warnings
from collections import Counter
import graphviz
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from PIL import Image
#import svgwrite
import scipy.ndimage
import numpy as np
import pandas as pd
import seaborn as sns
import math
from neat.graphs import required_for_output

import ML_frame

def plot_matrix(matrix, filename, en_w, en_h, classes=None, svg=False):
    h, w = matrix.shape
    #rescaled_matrix = (matrix - mean)/std
    rescaled_matrix = matrix / max(abs(np.amin(matrix)), abs(np.amax(matrix)))   
    print(np.amin(rescaled_matrix), np.amax(rescaled_matrix))
   
    img = np.empty([h,w,3], np.uint8) 
    #if svg:
    #    svg_doc = svgwrite.Drawing(filename=filename+'.svg', size=('{}px'.format(rescaled_matrix.shape[1]), '{}px'.format(rescaled_matrix.shape[0])))
    if classes is None:
        for i in xrange(h):
            for j in xrange(w):
                if matrix[i][j] >= 0.0:
                    r = 0
                    g = int(255*abs(rescaled_matrix[i][j]))
                    b = 0
                else:
                    r = 0
                    g = 0
                    b = int(255*abs(rescaled_matrix[i][j]))
                img[i][j] = [r,g,b]
                #if svg:
                #    svg_doc.add(svg_doc.rect(insert = (j, i), size = ("1px","1px"), fill="rgb({},{},{})".format(r,g,b)))
    else:
        classes = [ML_frame.convert_to_class(x) for x in classes]
        num_classes = len(set(classes))                    
        for i in xrange(h):
            for j in xrange(w):
                if matrix[i][j] >= 0.0:
                    if len(classes) == h:
                        r = int(50*classes[i]/num_classes)
                        g = int(255*abs(rescaled_matrix[i][j]))
                        b = 0
                    elif len(classes) == w:
                        r = int(50*classes[j]/num_classes)
                        g = int(255*abs(rescaled_matrix[i][j]))
                        b = 0                        
                else:
                    if len(classes) == h:
                        r = int(50*classes[i]/num_classes)
                        g = 0
                        b = int(255*abs(rescaled_matrix[i][j]))  
                    elif len(classes) == w:
                        r = int(50*classes[j]/num_classes)
                        g = 0
                        b = int(255*abs(rescaled_matrix[i][j]))
                img[i][j] = [r, g, b]
                #if svg:
                #    svg_doc.add(svg_doc.rect(insert = (j, i), size = ("1px","1px"), fill="rgb({},{},{})".format(r,g,b)))       
    
    new_img = scipy.ndimage.zoom(img, (en_w,en_h,1), order=0)  
    pilImage = Image.fromarray(new_img)
    pilImage.save(filename+'.png')
    #if svg:
    #    svg_doc.save()
   
def plot_stats(statistics, ylog=False, view=False, filename='out/avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return
    plt.figure()
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()
    plt.clf()
    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename='out/speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    plt.figure()
    species_sizes = statistics.get_species_sizes()
    first = [[0]*len(species_sizes[0])]
    species_sizes = first + species_sizes
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.clf()
    plt.close()

def plot_selection_history(history, filename):
    plt.figure()
    
    num_generations = len(history)
    n_his = []
    for g in history:
        values = [0] * len(g.values())
        for k in g.keys():
            values[(-k)-1] = g[k]
        n_his.append(values)
    
    remove = []
    for i in history[0].keys():
        all_0 = True
        for n in n_his:
            if n[(-i)-1] != 0:
                all_0 = False
        if all_0:
            remove.append((-i)-1)
    
    for n in n_his:    
        for index in sorted(remove, reverse=True):
            del n[index]        
    
    curves = np.array(n_his).T
    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)
    
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)    
    
    plt.title("Gene presence in population")
    plt.ylabel("Size per Genes")
    plt.xlabel("Generations")

    plt.savefig(filename)    
    
    plt.clf()
    plt.close()
    
    print('---\n{} genes were never selected!\n---'.format(len(remove)))
    with open(filename.replace('.png', '.txt'), 'w') as gh_file:
        gh_file.write('{} / {} ({:1.2f}) genes were never selected!\n---\n'.format(len(remove), len(history[0].keys()), float(len(remove)) / float(len(history[0].keys())))) 
        gh_file.write('Final generation total: {}\n'.format(np.array(history[-1].values()).sum()))             
        for k in history[-1].keys():
            if -(k+1) not in remove:
                gh_file.write('{:6d}: {}\n'.format(k, history[-1][k])) 
        gh_file.write('###########################################################\n') 
        gh_file.write('###########################################################\n')    

def required_nodes(config, genome, node_names=None, filename='out/nodes.txt', should_print=True): 
    if node_names is None:
        node_names = {}
    assert type(node_names) is dict
    
    used_inputs = set()
    for cg in genome.connections.values():
        if cg.enabled:
            used_inputs.add(cg.key[0])
    inputs = set()
    selected_inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))       
        if k in used_inputs:
            selected_inputs.add((k, name))
    
    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)

    connections = set()
    for k, v in zip(genome.connections.keys(), genome.connections.values()):
        if v.enabled:
            connections.add(k)
 
    used_nodes = set(required_for_output(inputs, outputs, connections))
 
    final_selected_inputs = set()
    for k in inputs:
        name = node_names.get(k, str(k))
        for n in used_nodes:
            if (k, n) in connections:
                final_selected_inputs.add((k, name))        
 
    with open(filename+'.txt', 'w') as nodes_file:
        if should_print:
            print('SELECTED INPUTS: ' + str(len(selected_inputs)))
        nodes_file.write('SELECTED INPUTS: ' + str(len(selected_inputs)) + '\n')
        if should_print:
            print(selected_inputs)
        nodes_file.write(str(selected_inputs))
        if should_print:
            print('FINAL SELECTED INPUTS: ' + str(len(final_selected_inputs)))
        nodes_file.write('\nFINAL SELECTED INPUTS: ' + str(len(final_selected_inputs)) + '\n')
        if should_print:
            print(final_selected_inputs)
        nodes_file.write(str(final_selected_inputs))
        if should_print:
            print('REQUIRED NODES: ' + str(len(used_nodes)))
        nodes_file.write('\nREQUIRED NODES: ' + str(len(used_nodes)) + '\n')
        if should_print:
            print(used_nodes)
        nodes_file.write(str(used_nodes))
        
    with open(filename+'_genes.txt', 'w') as genes_file:
        for gene in final_selected_inputs:
            genes_file.write(str(gene)+'\n')        
        
    return final_selected_inputs, used_nodes        

def plot_genes_stats(genes, num_nets, ax=None, filename='out/genes'):
    """"
    This function creates a bar plot from a counter.

    :param counter: This is a counter object, a dictionary with the item as the key
     and the frequency as the value
    :param ax: an axis of matplotlib
    :return: the axis wit the object in it
    """
    
    genes = [n[1] for n in genes]
    counter = Counter(genes)
    
    print('###############################################')
    print('GENES FREQUENCY')
    with open(filename+'.txt', 'w') as gene_file:
        s = sorted(counter, key=lambda x: counter[x], reverse=True)
        for k in s:
            print('[{:2d}] {:15s}: {}'.format(int(counter[k]), k, int(counter[k]) * '*'))
            gene_file.write('[{:2d}] {:15s}: {}\n'.format(int(counter[k]), k, int(counter[k]) * '*'))       
    
    for k in list(counter):
        if counter[k] < 2:
            del counter[k]
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    frequencies = counter.values()
    names = counter.keys()

    x_coordinates = np.arange(len(counter))
    ax.set_xlim([0, len(counter)])
    ax.bar(x_coordinates, map(lambda x: int(x), frequencies), color='g')
    ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(names))
     
    ax.set_ylim([0, num_nets+1])
    yres = 1
    start, stop = ax.get_ylim()
    ticks = np.arange(start, stop + yres, yres)
    ax.set_yticks(ticks)
    
    ax2 = ax.twinx()    
    ax2.set_ylim([0.0, 1.0])
    yres = 1.0 / float(num_nets)
    start, stop = ax2.get_ylim()
    ticks = np.arange(start, stop + yres, yres)
    ax2.set_yticks(ticks)
    
    ax.set_xlabel('Genes')
    ax.set_ylabel('Number of networks')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='left')
    plt.title("Selected genes")
    plt.savefig(filename+'.svg')
    plt.clf()
    plt.close()

def plot_genes_selection(num_genes, num_features, filename='out/genes_selected'):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.bar(np.arange(1, len(num_genes)+1, 1), num_genes)
    ax.set_xlim([1, len(num_genes)+1])
    ax.set_ylim([0, max(num_genes)])

    xres = 1
    start, stop = ax.get_xlim()
    ticks = np.arange(start, stop + xres, xres)
    ax.set_xticks(ticks)   

    yres = 1
    start, stop = ax.get_ylim()
    ticks = np.arange(start, stop + yres, yres)
    ax.set_yticks(ticks)   

    ax2 = ax.twinx()    
    ax2.set_ylim([0.0, max(num_genes)/float(num_features)])    

    ax.set_xlabel('Networks')
    ax.set_ylabel('Number of genes')
    ax2.set_ylabel('Proportion')    
    
    plt.tight_layout()
    plt.setp(ax.get_xticklabels(), horizontalalignment='center')
    plt.title("Number of selected genes")
    plt.savefig(filename+'.svg')
    plt.clf()
    plt.close()    

def plot_selected_genes_expression(dataset, genes, classes, classes_labels, filename):
    indexes = [abs(i[0])-1 for i in genes]
    names   = [n[1] for n in genes]
    indexes, names = zip(*sorted(zip(indexes, names)))     
    classes = [classes_labels[ML_frame.convert_to_class(c)] for c in classes]
    data = dataset[:, indexes]
    
    sns.set(style="white")
    I = pd.Index(classes, name="Samples")
    C = pd.Index(names, name="Genes")
    d = pd.DataFrame(data=data, index=I, columns=C).transpose()
    
    print(genes)
    print(indexes)
    print(names)
    print(dataset.shape)
    print(data.shape)
    #print(d)
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(40,20))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(h_neg=190, h_pos=100, l=50, center='dark', as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(d, cmap=cmap, vmin=-0.7, vmax=0.7, center=0.0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(filename+'.svg')
    plt.clf()
    plt.close()  
     
    with open(filename+'.txt', 'w') as df:
        df.write(d.to_csv())                  

def save_gene_selection(data_file, genes, filename):
    #indexes = set([abs(i[0])+3 for i in genes] + [0, 1, 2])
    indexes = [abs(i[0])-1+3 for i in genes] + [0, 1, 2]
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

def plot_genes_correlation(dataset, genes, filename):
    indexes = [abs(i[0])-1 for i in genes]
    names   = [n[1] for n in genes]
    indexes, names = zip(*sorted(zip(indexes, names)))    
    data = dataset[:, indexes]
    
    sns.set(style="white")
    d = pd.DataFrame(data=data, columns=names)
    # Compute the correlation matrix
    corr = d.corr()
    #print(corr)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(filename+'.svg')
    plt.clf()
    plt.close()               

def boxplot(dataset, filename, label=False, genes=None, classes=None, classes_names=None):
    indexes = [abs(i[0])-1 for i in genes]
    names   = [n[1] for n in genes]
    indexes, names = zip(*sorted(zip(indexes, names)))    
    data = dataset[:, indexes]    
    d = pd.DataFrame(data=data, columns=names)    
    if classes is not None:
        labels = []
        if classes_names is None:
            classes_names = [str(k) for k in classes[0]]
        for sample in classes:
            labels.append(classes_names[ML_frame.convert_to_class(sample)])
        labels = np.array(labels)
        d = d.assign(Class=labels) 
        melted_d = pd.melt(d, 
                    id_vars=["Class"], # Variables to keep
                    var_name="Genes") # Name of melted variable               
        ax = sns.boxplot(data=melted_d, x='Genes', y='value', hue='Class', dodge=True, palette="Set3")
    else:
        ax = sns.boxplot(data=d, palette="Set3")
    if len(indexes) > 4:
        plt.setp(ax.get_xticklabels(), rotation=90) 
        plt.subplots_adjust(bottom=0.3)
    ax.set(xlabel='Genes', ylabel='Expression')   
    filename = filename+'.svg'
    plt.savefig(filename)
    plt.clf()
    plt.close()        

def boxplot2(dataset1, dataset2, classes1, classes2, filename, label=False, genes=None, classes_names=None):
    indexes = [abs(i[0])-1 for i in genes]
    names   = [n[1] for n in genes]
    indexes, names = zip(*sorted(zip(indexes, names)))    
    data1 = dataset1[:, indexes]    
    d1 = pd.DataFrame(data=data1, columns=names)
    data2 = dataset2[:,indexes]     
    d2 = pd.DataFrame(data=data2, columns=names)

    if classes1 is not None:
        labels = []
        if classes_names is None:
            classes_names = [str(k) for k in classes1[0]]
        for sample in classes1:
            labels.append(classes_names[ML_frame.convert_to_class(sample)] + ' Tr')
        labels = np.array(labels)
        d1 = d1.assign(Class=labels) 
    
    if classes2 is not None:
        labels = []
        if classes_names is None:
            classes_names = [str(k) for k in classes2[0]]
        for sample in classes2:
            labels.append(classes_names[ML_frame.convert_to_class(sample)] + ' Te')
        labels = np.array(labels)
        d2 = d2.assign(Class=labels)                
    
    d = pd.concat([d1, d2])
    melted_d = pd.melt(d, 
                id_vars=["Class"], # Variables to keep
                var_name="Genes") # Name of melted variable    
    
    ax = sns.boxplot(data=melted_d, x='Genes', y='value', hue='Class', dodge=True, palette="Set3")               
    if len(indexes) > 4:
        plt.setp(ax.get_xticklabels(), rotation=90) 
        plt.subplots_adjust(bottom=0.3)   
    ax.set(xlabel='Genes', ylabel='Expression')
    filename = filename+'.svg'
    plt.savefig(filename)
    plt.clf()
    plt.close() 

def plot_distribution(dataset, filename, label=False, genes=None, classes=None, classes_names=None):
    show_hist = False
    colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y', 'grey', 'orange', 'beige', 'pink', 'gold']
    if classes is not None:
        linestyles = ['-', '--', '-.', ':']
        if classes_names is None:
            classes_names = [str(k) for k in classes[0]]    
        if genes is not None:
            indexes = [abs(i[0])-1 for i in genes]
            names   = [n[1] for n in genes]   
            indexes, names = zip(*sorted(zip(indexes, names)))    
            data = dataset[:, indexes]         
            for gene in xrange(data.shape[1]):
                gene_expression = data[:,gene]
                c = []
                for o in xrange(classes.shape[1]):
                    c.append([])
                for sample in zip(classes, gene_expression):
                     c[ML_frame.convert_to_class(sample[0])].append(sample[1])            
            
            
                c = [np.array(a) for a in c]
                i = 0    
                for cs in c:
                    if label:
                       sns.distplot(cs, hist=show_hist, label=names[gene] + ' ' + classes_names[i], kde_kws={'color': colors[gene%len(colors)], 'linestyle':linestyles[i%len(linestyles)]}).set(xlabel='Gene expression', ylabel='Distribution')
                    else:
                       sns.distplot(cs, hist=show_hist, kde_kws={'color': colors[gene%len(colors)], 'linestyle':linestyles[i%len(linestyles)]}).set(xlabel='Gene expression', ylabel='Distribution')
                    i += 1       
            filename = filename+'.svg'
        
        else:
            for gene in xrange(dataset.shape[1]):
                gene_expression = dataset[:,gene]
                c = []
                for o in xrange(classes.shape[1]):
                    c.append([])
                for sample in zip(classes, gene_expression):
                     c[ML_frame.convert_to_class(sample[0])].append(sample[1])                
                c = [np.array(a) for a in c]
                i = 0
                for cs in c:
                    if label:
                       sns.distplot(cs, hist=show_hist, label=names[gene], kde_kws={'color': colors[gene%len(colors)], 'linestyle':linestyles[i%len(linestyles)]}).set(xlabel='Gene expression', ylabel='Distribution')
                    else:
                       sns.distplot(cs, hist=show_hist, kde_kws={'color': colors[gene%len(colors)], 'linestyle':linestyles[i%len(linestyles)]}).set(xlabel='Gene expression', ylabel='Distribution')
                    i += 1
    
    else:
        if genes is not None:
            indexes = [abs(i[0])-1 for i in genes]
            names   = [n[1] for n in genes]   
            indexes, names = zip(*sorted(zip(indexes, names)))    
            data = dataset[:, indexes]         
            for gene in xrange(data.shape[1]):
                if label:
                   sns.distplot(data[:,gene], hist=show_hist, label=names[gene], kde_kws={'color': colors[gene%len(colors)]}).set(xlabel='Gene expression', ylabel='Distribution')
                else:
                   sns.distplot(data[:,gene], hist=show_hist, kde_kws={'color': colors[gene%len(colors)]}).set(xlabel='Gene expression', ylabel='Distribution')
            filename = filename+'.svg'       
        else:
            for gene in xrange(dataset.shape[1]):
                sns.distplot(dataset[:,gene], hist=show_hist, kde_kws={'color': colors[gene%len(colors)]}).set(xlabel='Gene expression', ylabel='Distribution')   
                                 
    plt.savefig(filename)
    plt.clf()
    plt.close()

def draw_net(config, genome, view=False, filename='out/net', node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. 
    visualize.draw_net(config, winner, True, node_names=node_names)"""
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.1',
        'width': '0.1'}

    #dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)
    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs, engine='neato')
    dot.attr(overlap='false')
    dot.attr(splines = 'true')

    used_inputs = set()
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            used_inputs.add(cg.key[0])

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box'}
        input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
        
        if k in used_inputs:
            dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))
                print(cg.in_node_id)

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            #if input not in used_nodes or output not in used_nodes:
            #    continue
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)
    return dot

def draw_required_net(config, genome, view=False, filename='out/reqnet', node_names=None, show_disabled=False, node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. 
    visualize.draw_net(config, winner, True, node_names=node_names)"""
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.1',
        'width': '0.1'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs, engine='neato')
    dot.attr(overlap='false')
    dot.attr(splines = 'true')

    used_inputs = set()
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            used_inputs.add(cg.key[0])

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)

    connections = set()
    for k, v in zip(genome.connections.keys(), genome.connections.values()):
        if v.enabled or show_disabled:
            connections.add(k)

    used_nodes = set(required_for_output(inputs, outputs, connections))

    selected_inputs = set()
    for k in inputs:
        name = node_names.get(k, str(k))
        for n in used_nodes:
            if (k, n) in connections:
                name = node_names.get(k, str(k))
                input_attrs = {'style': 'filled',
                               'shape': 'box'}
                input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
                dot.node(name, _attributes=input_attrs)               
                selected_inputs.add(k) 

    for k in outputs:
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')
        dot.node(name, _attributes=node_attrs) 

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            if (input not in selected_inputs and input not in used_nodes) or output not in used_nodes:
                continue
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)
    return dot    
