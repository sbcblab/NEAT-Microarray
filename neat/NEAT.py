# BRUNO IOCHINS GRISCI

from __future__ import print_function

from shutil import copyfile
import sys
import os
import math
import random
import neatB
import neatB.math_util
from neat.six_util import iteritems, itervalues
from collections import Counter
import numpy as np
from scipy import stats

import visualize
import eval_functions
import log_error
import ML_frame
from microarray_reader import MAR
import mrmr2gct

np.set_printoptions(precision=3, threshold=50, suppress=True)

################################################################################

def eval_genome(genome, config):
    global TR_INPUTS
    global TR_OUTPUTS
    global SELECTED
    global decay    
    global ef        
  
    if -1 not in SELECTED: 
        inp = []
        out = []         
        for s in SELECTED:
            inp.append( TR_INPUTS[s])
            out.append(TR_OUTPUTS[s])
            '''for noise in xrange(1):
                inp.append([np.random.normal(i, 0.5*0.01) for i in TR_INPUTS[s]])
                out.append(TR_OUTPUTS[s])'''
        try:
            if genome.parent1_fit == None and genome.parent2_fit == None:
                inheritance = 0.0
            elif genome.parent1_fit == None:
                inheritance = genome.parent2_fit * decay
            elif genome.parent2_fit == None:
                inheritance = genome.parent1_fit * decay
            else:
                inheritance = ((genome.parent1_fit + genome.parent2_fit) / 2.0) * decay
            fit = ef(genome, config, inp, out) + inheritance #- eval_functions.features_selected(genome, config, 0.5)[0]            
            #fit = eval_functions.cross_entropy(genome, config, inp, out) + inheritance
        except OverflowError:
            fit = -99999.99
            print('Overflow error')
    else:
        try:
            #fit = eval_functions.cross_entropy(genome, config, TR_INPUTS, TR_OUTPUTS)# - eval_functions.features_selected(genome, config, 0.5)      
            fit = ef(genome, config, TR_INPUTS, TR_OUTPUTS) - eval_functions.L2_reg(genome, 0.5, len(TR_INPUTS))
        except OverflowError:
            fit = -99999.99
            print('Overflow error')
    return fit

def sort_genomes(population, config, tr_inputs, tr_outputs):
    genomes_info = []
    for key in population:
        genomes_info.append([population[key], population[key].fitness])
    genomes_info.sort(key=lambda x: x[1], reverse=True)
    return genomes_info
    
def print_population(population, config, node_names, tr_inputs, tr_outputs, filepath):
    with open(filepath + 'last_pop.txt', 'w') as pop_file:
        i = 1
        for genome, fit in sort_genomes(population, config, tr_inputs, tr_outputs)[0:100]:
            visualize.draw_net(config, genome, view=False, node_names=node_names, filename=filepath + 'pop/net_{:03d}_{}'.format(i, genome.key))
            size = genome.size()
            sel, ngenes  = eval_functions.features_selected(genome, config, 1.0)
            pop_file.write('{:05d}, {:03d}, {:03d}, {:03d}, {}\n'.format(genome.key, ngenes, size[0], size[1], genome.fitness))
            i += 1               

'''def ensemble(p, config, number, node_names, save_dir, tr_inputs, tr_outputs, te_inputs=None, te_outputs=None, classes_labels=None):
    save_dir = save_dir + 'ensemble/'
    population = sort_genomes(p.population, config, tr_inputs, tr_outputs)[0:number]
    fitness_computed = []
    train_acc = []
    tests_acc = []
    all_genes = []
    for ind in xrange(len(population)): 
        visualize.draw_required_net(config, population[ind], view=False, node_names=node_names, filename=save_dir+'net'+str(ind))
        genes, nodes = visualize.required_nodes(config, population[ind], node_names=node_names, filename=save_dir+'nodes'+str(ind), should_print=False)
        for g in genes:
            all_genes.append(g)
        #fitness_computed.append(eval_functions.imbalanced_cross_entropy(population[ind], config, tr_inputs, tr_outputs))
        with open(save_dir+'genome'+str(ind)+'.txt', 'w') as genome_file:
            genome_file.write('Genome:\n{!s}'.format(population[ind]))
        with open(save_dir+'voters_genes.txt', 'a') as vg_file:
            vg_file.write(str(len(set(all_genes)))+'\n')
    #with open(save_dir + 'fitness_computed.txt', 'w') as fitness_file:
    #    for f in fitness_computed:
    #        fitness_file.write(str(f) + '\n')  
    with open(save_dir + 'fitness.txt', 'w') as fitness_file:
        for ind in population:
            fitness_file.write(str(ind.fitness) + '\n')
    
    all_genes = set(all_genes)       
    visualize.plot_selected_genes_expression(tr_inputs, all_genes, tr_outputs, classes_labels, save_dir+'tr_sel_genes_exp')
    visualize.plot_genes_correlation(tr_inputs, all_genes, save_dir+'tr_genes_correlation')
    visualize.save_gene_selection(tr_data_file, all_genes, save_dir+'tr_selection')   
    
    if te_inputs is not None and te_outputs is not None:
        visualize.plot_selected_genes_expression(te_inputs, all_genes, te_outputs, classes_labels, save_dir+'te_sel_genes_exp')
        visualize.plot_genes_correlation(te_inputs, all_genes, save_dir+'te_genes_correlation')
    
    #size = len(population)
    #weighted_population = []
    #for i in xrange(size+1):
    #    weighted_population = weighted_population + (size-i)*population[i:i+1]

    return population'''                         

def convert_to_class(entry):
    if type(entry) is not list:
        entry = entry.tolist()
    return entry.index(max(entry))

'''def mrmr(save_dir, n_features, features_names, tr_inputs, tr_outputs, te_inputs=None, ):
    global tr_data_file
    mrmr_file = save_dir + 'mrmr.txt'
    with open(save_dir+'train.csv', 'w') as csv_file:
        header = 'class'
        for name in features_names:
            header = header + ', ' + name
        header = header + '\n'
        csv_file.write(header)
        classes = []
        for to in tr_outputs:
            classes.append(convert_to_class(to) + 1)
        for d in zip(classes, tr_inputs):
            line = str(d[0])
            for v in d[1]:
                line = line + ', ' + str(v)
            line = line + '\n'
            csv_file.write(line) 
    
    os.system("./mrmr -i {} -n {} -s {} -v {} > {}".format(save_dir+'train.csv', n_features, len(tr_inputs), len(features_names), mrmr_file))
    #os.system("/home/bruno/./mrmr -i {} -n {} -s {} -v {} > {}".format(save_dir+'train.csv', n_features, len(tr_inputs), len(features_names), mrmr_file))
    
    order = []
    fea   = []
    name  = [] 
    score = []
    with open(mrmr_file, 'r') as mf:
        while '*** mRMR features ***' not in mf.readline():
            pass
        line = mf.readline()
        while line.strip():
            line = mf.readline()
            if line.strip():
                l = line.split()
                order.append(int(l[0]))
                fea.append(int(l[1])-1)
                name.append(l[2])
                score.append(float(l[3]))   
    
    mrmr_features_names = []
    mrmr_tr_inputs = None
    mrmr_te_inputs = None
        
    for f in sorted(fea):
        mrmr_features_names.append(features_names[f])
   
    not_selected = np.arange(len(features_names))
    not_selected = np.delete(not_selected, np.array(sorted(fea)))
    print(fea)
    print(not_selected)
    mrmr_tr_inputs = np.delete(tr_inputs, not_selected, 1)
    if te_inputs is not None:   
        mrmr_te_inputs = np.delete(te_inputs, not_selected, 1)
    
    mrmr_gct_file = mrmr2gct.save_gene_selection(tr_data_file, fea, mrmr_file.replace('.txt', ''))
    
    with open(save_dir+'mrmr_selection.txt', 'w') as mrmrsel_file:
        mrmrsel_file.write("\n".join(name for name in mrmr_features_names))
        
    print(len(mrmr_tr_inputs))
    print(len(mrmr_tr_inputs[0]))
    print(name)
    print(mrmr_features_names)
    
    tr_mean = np.mean(mrmr_tr_inputs, axis=0)
    tr_std  = np.std(mrmr_tr_inputs, axis=0)
    mrmr_tr_inputs = MAR.standardize_dataset(mrmr_tr_inputs, tr_mean, tr_std)
    if te_inputs is not None:
        mrmr_te_inputs = MAR.standardize_dataset(mrmr_te_inputs, tr_mean, tr_std)
    
    return mrmr_tr_inputs, mrmr_te_inputs, mrmr_features_names, mrmr_gct_file'''                    

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
        
        pv = 1.0
        try:
            pv = stats.kruskal(*classes)[1]
        except ValueError:
            pass
        genes_pvalues.append(pv)

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
    #probs = neatB.math_util.softmax(c * np.log(pvalues))
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
        print(1, np.sum(sor[:1]))
        print(10, np.sum(sor[:10]))
        print(20, np.sum(sor[:20]))        
        print(100, np.sum(sor[:100])) 
        print(500, np.sum(sor[:500]))         
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
        if pvalues[i] < threshold:
            sel_p.append(pvalues[i])
            sel_fn.append(features_names[i])
            fea.append(i)

    if len(sel_p) <= 1:
        print("TRYING 2x threshold for KW!")
        sel_p =  []
        sel_fn = []
        fea =    []
        for i in xrange(len(pvalues)):
            if pvalues[i] < threshold*2.0:
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

def save_database_stats(tr_inputs, te_inputs, tr_outputs, te_outputs, save_dir):
    with open(save_dir + 'db_stats.txt', 'w') as db_stats:
        db_stats.write('TRAIN size: {}\n'.format(tr_inputs.shape))
        if te_inputs is not None:
            db_stats.write('TEST  size: {}\n'.format(te_inputs.shape))  
        if len(tr_inputs) > 7:
            k2, p = stats.normaltest(tr_inputs, axis=0)
            #This function tests the null hypothesis that a sample comes from a normal distribution. 
            #It is based on D'Agostino and Pearson's [1], [2] test 
            #that combines skew and kurtosis to produce an omnibus test of normality.
            db_stats.write('TRAIN norm: {:5d}, {}\n'.format(np.sum(p >= 1e-3), p)) 
        else:
            db_stats.write('TRAIN norm: less than 8 samples\n')   
        if te_inputs is not None:
            if len(te_inputs) > 7:  
                k2, p = stats.normaltest(te_inputs, axis=0)
                db_stats.write('TEST  norm: {:5d}, {}\n'.format(np.sum(p >= 1e-3), p)) 
            else:
                db_stats.write('TEST  norm: less than 8 samples\n')                                  
        
        if te_inputs is not None:
            genes_pvalues = []
            for gene in xrange(tr_inputs.shape[1]):
                tr_gene_expression = tr_inputs[:,gene]
                te_gene_expression = te_inputs[:,gene]
                pv = 1.0
                try:
                    pv = stats.kruskal(tr_gene_expression, te_gene_expression)[1]
                except ValueError:
                    pass
                genes_pvalues.append(pv)
            genes_pvalues = np.array(genes_pvalues)
            db_stats.write('KW_ALL:     {:5d}, {}\n'.format(np.sum(genes_pvalues < 0.01), genes_pvalues))            

            genes_tr = []
            for gene in xrange(tr_inputs.shape[1]):
                gene_expression = tr_inputs[:,gene]
                tr_classes = []
                for o in xrange(tr_outputs.shape[1]):
                    tr_classes.append([])
                for sample in zip(tr_outputs, gene_expression):
                     tr_classes[convert_to_class(sample[0])].append(sample[1])
                genes_tr.append(tr_classes)

            genes_te = []
            for gene in xrange(te_inputs.shape[1]):
                gene_expression = te_inputs[:,gene]
                te_classes = []
                for o in xrange(te_outputs.shape[1]):
                    te_classes.append([])
                for sample in zip(te_outputs, gene_expression):
                     te_classes[convert_to_class(sample[0])].append(sample[1])
                genes_te.append(te_classes)
            
            genes_C00 = []
            genes_C11 = []    
            genes_C01 = []
            genes_C10 = [] 
                  
            genes_TR  = []
            genes_TE  = []     
              
            for gtr, gte in zip(genes_tr, genes_te):
                
                pv = 1.0
                try:
                    pv = stats.kruskal(gtr[0], gte[0])[1]
                except ValueError:
                    pass
                genes_C00.append(pv)

                pv = 1.0
                try:
                    pv = stats.kruskal(gtr[1], gte[1])[1]
                except ValueError:
                    pass
                genes_C11.append(pv)

                pv = 1.0
                try:
                    pv = stats.kruskal(gtr[0], gte[1])[1]
                except ValueError:
                    pass
                genes_C01.append(pv)

                pv = 1.0
                try:
                    pv = stats.kruskal(gtr[1], gte[0])[1]
                except ValueError:
                    pass
                genes_C10.append(pv)

                pv = 1.0
                try:
                    pv = stats.kruskal(gtr[0], gtr[1])[1]
                except ValueError:
                    pass
                genes_TR.append(pv)                
                
                pv = 1.0
                try:
                    pv = stats.kruskal(gte[0], gte[1])[1]
                except ValueError:
                    pass
                genes_TE.append(pv)
                                                         
            genes_C00 = np.array(genes_C00)
            genes_C11 = np.array(genes_C11)
            genes_C01 = np.array(genes_C01)
            genes_C10 = np.array(genes_C10)
            
            genes_TR = np.array(genes_TR)
            genes_TE = np.array(genes_TE)            
                        
            db_stats.write('KW_C00:     {:5d}, {}\n'.format(np.sum(genes_C00 < 0.01), genes_C00))    
            db_stats.write('KW_C11:     {:5d}, {}\n'.format(np.sum(genes_C11 < 0.01), genes_C11))   
            db_stats.write('KW_C01:     {:5d}, {}\n'.format(np.sum(genes_C01 < 0.01), genes_C01))    
            db_stats.write('KW_C10:     {:5d}, {}\n'.format(np.sum(genes_C10 < 0.01), genes_C10))  
            
            db_stats.write('KW_TR:      {:5d}, {}\n'.format(np.sum(genes_TR < 0.01), genes_TR))    
            db_stats.write('KW_TE:      {:5d}, {}\n'.format(np.sum(genes_TE < 0.01), genes_TE))                         
        
        db_stats.write('TRAIN mean: {}\n'.format(np.mean(tr_inputs, axis=0)))
        db_stats.write('TRAIN  std: {}\n'.format(np.std(tr_inputs, axis=0))) 
        if te_inputs is not None:        
            db_stats.write('TEST  mean: {}\n'.format(np.mean(te_inputs, axis=0)))
            db_stats.write('TEST   std: {}\n'.format(np.std(te_inputs, axis=0)))
        
        db_stats.write('TRAIN medi: {}\n'.format(np.median(tr_inputs, axis=0)))
        if te_inputs is not None:        
            db_stats.write('TEST  medi: {}\n'.format(np.median(te_inputs, axis=0)))        
        db_stats.write('TRAIN MEAN: {:4.4f} +- {:4.4f}\n'.format(np.mean(tr_inputs), np.std(tr_inputs))) 
        if te_inputs is not None:        
            db_stats.write('TEST  MEAN: {:4.4f} +- {:4.4f}\n'.format(np.mean(te_inputs), np.std(te_inputs))) 
        db_stats.write('TRAIN MEDI: {:4.4f}\n'.format(np.median(tr_inputs))) 
        if te_inputs is not None:        
            db_stats.write('TEST  MEDI: {:4.4f}\n'.format(np.median(te_inputs)))                                       
        
        db_stats.write('TRAIN  min: {}\n'.format(tr_inputs.min(0)))
        if te_inputs is not None:        
            db_stats.write('TEST   min: {}\n'.format(te_inputs.min(0)))
    
        db_stats.write('TRAIN  max: {}\n'.format(tr_inputs.max(0)))
        if te_inputs is not None:        
            db_stats.write('TEST   max: {}\n'.format(te_inputs.max(0)))
            db_stats.write('MIN   DIFF: {:4.4f}, {}\n'.format(np.mean(np.absolute(tr_inputs.min(0) - te_inputs.min(0))), np.absolute(tr_inputs.min(0) - te_inputs.min(0))))        
            db_stats.write('MAX   DIFF: {:4.4f}, {}\n'.format(np.mean(np.absolute(tr_inputs.max(0) - te_inputs.max(0))), np.absolute(tr_inputs.max(0) - te_inputs.max(0))))        
                       
def run(nothing, tr_inputs, tr_outputs, te_inputs=None, te_outputs=None, tr_names=None, te_names=None, features_names=None, classes_labels=None, save_dir='', num_cores=1, num_gen=1):
    # Assign new values to global variables
    # Function eval_genome must receive the dataset arguments as global variables
    global TR_INPUTS
    global TR_OUTPUTS
    global SELECTED
    global GENES
    global NUM_GENES
    global tr_data_file 
    global config_path
    global decay
    global ef

    if not os.path.exists(save_dir+'genes/'):
        os.makedirs(save_dir+'genes/')

    ############################################################################

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

    save_database_stats(tr_inputs, te_inputs, tr_outputs, te_outputs, save_dir)
    '''visualize.plot_distribution(tr_inputs, save_dir+'genes/dist_train')
    visualize.plot_distribution(tr_inputs, save_dir+'genes/dist_train_cl', classes=tr_outputs, classes_names=classes_labels)
    if te_inputs is not None:
        visualize.plot_distribution(te_inputs, save_dir+'genes/dist_test')
        visualize.plot_distribution(te_inputs, save_dir+'genes/dist_test_cl', classes=te_outputs, classes_names=classes_labels)'''
    
    '''tr_mean = np.mean(tr_inputs, axis=0)
    tr_std  = np.std(tr_inputs, axis=0)
    tr_inputs = MAR.standardize_dataset(tr_inputs, tr_mean, tr_std)
    if te_inputs is not None:
        te_inputs = MAR.standardize_dataset(te_inputs, tr_mean, tr_std)'''
     
    '''tr_min = tr_inputs.min(0)
    tr_ptp = tr_inputs.ptp(0)    
    tr_inputs = 2.0 * ((tr_inputs - tr_min)/tr_ptp) - 1.0
    if te_inputs is not None:
        te_inputs = 2.0 * ((te_inputs - tr_min)/tr_ptp) - 1.0'''
        
    tr_mean = np.mean(tr_inputs, axis=0)
    tr_ptp = tr_inputs.ptp(0)       
    tr_inputs = (tr_inputs - tr_mean)/tr_ptp
    if te_inputs is not None:
        te_inputs = (te_inputs - tr_mean)/tr_ptp
    
    print(tr_inputs.shape)
    if te_inputs is not None:
        print(te_inputs.shape)
        print('Underflow in test normalization: {:3d}'.format(np.sum(te_inputs.min(0) < -1.0)))
        print('Overflow  in test normalization: {:3d}'.format(np.sum(te_inputs.max(0) > 1.0)))            
    
    save_database_stats(tr_inputs, te_inputs, tr_outputs, te_outputs, save_dir+'n')

    '''visualize.plot_distribution(tr_inputs, save_dir+'genes/dist_ntrain')
    visualize.plot_distribution(tr_inputs, save_dir+'genes/dist_ntrain_cl', classes=tr_outputs, classes_names=classes_labels)    
    if te_inputs is not None:
        visualize.plot_distribution(te_inputs, save_dir+'genes/dist_ntest')
        visualize.plot_distribution(te_inputs, save_dir+'genes/dist_ntest_cl', classes=te_outputs, classes_names=classes_labels)'''
        
    TR_INPUTS  = tr_inputs
    TR_OUTPUTS = tr_outputs
    
    ############################################################################

    print('Creating a configuration file!')
    # Load configuration.
    with open(config_path, 'r') as config_file:
        con = config_file.read()
        pop_index = con.find("pop_size              = ") + len("pop_size              = ")
        pop_size = float(con[pop_index:con.find("reset_on_extinction   = False")].replace('\n', ''))
        fraction = (1.0/(pop_size)) / float(tr_outputs.shape[1])
        con = con.replace("max_stagnation       = 0", "max_stagnation       = {}".format(int(0.1 * num_gen)))
        con = con.replace("elitism            = 0", "elitism            = {}".format(int(0.1 * pop_size)))
        con = con.replace("min_species_size = 0", "min_species_size = {}".format(int(0.25 * pop_size)))
        con = con.replace("num_inputs              = 0", "num_inputs              = {}".format(tr_inputs.shape[1]))
        #con = con.replace("num_outputs             = 0", "num_outputs             = {}".format(tr_outputs.shape[1]))
        con = con.replace("num_outputs             = 0", "num_outputs             = {}".format(1))   
        con = con.replace("initial_connection      = partial_nodirect 0.0", "initial_connection      = partial_nodirect {}".format(fraction))
        with open(save_dir + 'c-n', 'w') as cn_file:
            cn_file.write(con)
    
    config = neatB.Config(neatB.DefaultGenome, neatB.DefaultReproduction,
                         neatB.DefaultSpeciesSet, neatB.DefaultStagnation,
                         save_dir + 'c-n', num_gen, probs)
    print(config.pop_size)
    print(config.genome_config.initial_connection, config.genome_config.connection_fraction)
    print(config.genome_config.num_inputs, config.genome_config.num_outputs)

    #print(config.probs)

    #if config.genome_config.num_outputs != len(tr_outputs[0]):
    #    raise ValueError('Number of outputs is not the number of classes!')

    ############################################################################

    print('Initializing!')

    selection_history = []

    # Create the population, which is the top-level object for a NEAT run.
    p = neatB.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neatB.StdOutReporter(True))
    stats = neatB.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(generation_interval=int(num_gen/2), filename_prefix=save_dir+'neat-checkpoint-'))

    # Run for up to n generations.
    #pe = neat.ParallelEvaluator(num_cores, eval_genome)
    
    num_classes = len(TR_OUTPUTS[0])
    outs = np.array(TR_OUTPUTS)
    winner = None
    last_winner = None
    kqrs = False
    window_size = 0
    tabu_max = 0
    
    if kqrs:
        window_size = num_classes - 1
        
        I = []
        for i in xrange(num_classes):
            w = np.where(outs[:,i] == 1)[0].tolist()
            I.append(w)    

        smallest_class = 1000000
        for i in I:
            if len(i) < smallest_class:
                smallest_class = len(i)
        while smallest_class - 1 <= window_size:
            window_size = window_size - 1
        if window_size < 1:
            window_size = 1

        S = []
        for i in I:
            S.append(random.sample(i, window_size))
    
        tabu_max = (smallest_class / (window_size * 2.0)) * num_classes
    
    print('Running NEAT with LEEA = {}, window = {}, and tabu = {}!'.format(kqrs, window_size, tabu_max))
    
    TABU = []
    
    ###
    node_names = {-(k+1): v for k, v in enumerate(features_names)}
    node_names.update({k: v for k, v in enumerate(classes_labels)})
    ###
    
    gene_selection_history = []
    gene_selection_history.append({-(k+1): 0 for k, v in enumerate(features_names)})
    BEST_NN_GENES = []
    BEST_NN_GENES.append([])
    
    '''if kqrs:
        SELECTED = []
        for i in I:
            SELECTED = SELECTED + random.sample([item for item in i if item not in SELECTED], window_size)   
        for i in I:
            SELECTED = SELECTED + random.sample([item for item in i if item not in SELECTED], window_size)'''

    with open(save_dir + 'gene_number_track.txt', 'a') as gtrack_file:
        gtrack_file.write('ALL, CURRENT, NEW, GONE\n')
    
    all_genes  = set()
    curr_genes = set()
    last_genes = set()
    for generation in xrange(num_gen):
        if kqrs:
            SELECTED = []
            if generation < num_gen - 1:
            #if True:
                '''for s, i in zip(S, I):
                    s.append(random.choice([item for item in i if item not in s]))
                    s.pop(0)
                for s in S:                  
                    SELECTED = SELECTED + s'''
                for i in I:
                    #SELECTED = SELECTED + random.sample(i, window_size)
                    sel = random.sample([item for item in i if item not in TABU and item not in SELECTED], window_size)
                    SELECTED = SELECTED + sel
                    #for se in sel:
                    #    TABU.append(se)
                #while len(TABU) > tabu_max:
                #    TABU.pop(0)
                '''for s in xrange(window_size * num_classes):
                    SELECTED.pop(0)  '''
            else:
                SELECTED = []
                SELECTED.append(-1)
        else:
            SELECTED = []
            SELECTED.append(-1)                
             
        print('===============================================================')
        print(TABU)
        print(SELECTED)        
        #print('CAP: {}'.format(config.conn_add_prob))   
        selection_history.append(SELECTED)  
        pe = neatB.ParallelEvaluator(num_cores, eval_genome)  
        last_winner = winner
        winner = p.run(pe.evaluate, 1)
        
        bnng = []
        for c in winner.connections:
            if c[0] < 0 and c[0] not in bnng and winner.connections[c].enabled == True:
                bnng.append(c[0])
        BEST_NN_GENES.append(bnng)        
        
        last_genes = curr_genes - set()
        curr_genes = set()
        gs = {-(k+1): 0 for k, v in enumerate(features_names)}
        for key in p.end_population:
            saved = []
            for c in p.end_population[key].connections:
                if c[0] < 0 and c[0] not in saved and p.end_population[key].connections[c].enabled == True:
                    saved.append(c[0])
                    curr_genes.add(c[0])
                    gs[c[0]] = gs[c[0]] + 1    
        gene_selection_history.append(gs)
        
        hey_genes = curr_genes - last_genes
        bye_genes = last_genes - curr_genes
        all_genes = all_genes.union(curr_genes)
        
        with open(save_dir + 'gene_number_track.txt', 'a') as gtrack_file:
            gtrack_file.write('{}, {}, {}, {}\n'.format(len(all_genes), len(curr_genes), len(hey_genes), len(bye_genes)))
        
        ###
        if last_winner is not None:
            if last_winner.key != winner.key:
                visualize.draw_net(config, winner, view=False, node_names=node_names, filename=save_dir+'nets/net'+str(winner.key))
        ###
        
        #print('\nBest genome:\n{!s}'.format(winner))
        #print('CAP: {}'.format(1.0 - (winner.generation / config.num_gen )))
        #print('CDP: {}'.format(winner.generation / (config.num_gen * 2.0) ))
        #true_winner = sort_genomes(p.population, config, tr_inputs, tr_outputs)[0]
        print('{}: {} // {}: {} - {}: {}'.format(winner.key, winner.fitness, winner.parent1_key, winner.parent1_fit, winner.parent2_key, winner.parent2_fit))
        
        with open(save_dir + 'offspring.txt', 'a') as offspring_file:
            method = ''
            if winner.parent2_key is None:
                method = 'elitism'
            elif winner.parent1_key == winner.parent2_key:
                method = 'mutation'
            elif winner.parent1_key != winner.parent2_key:
                method = 'crossover'
            else:
                method = '???'
            offspring_file.write(method + '\n')                
        with open(save_dir + 'inheritance_history_file.txt', 'a') as inheritance_history_file:
            if winner.parent1_fit is None and winner.parent2_fit is None:
                inheritance = 0.0
            elif winner.parent1_fit is None:
                inheritance = winner.parent2_fit #* decay
            elif winner.parent2_fit is None:
                inheritance = winner.parent1_fit #* decay
            else:
                inheritance = ((winner.parent1_fit + winner.parent2_fit) / 2.0) #* decay
            inheritance_history_file.write(str(inheritance) + '\n')
 
        with open(save_dir + 'parent1_inheritance_history_file.txt', 'a') as parent1_inheritance_history_file:
            parent1_inheritance_history_file.write(str(winner.parent1_fit) + '\n')

        with open(save_dir + 'parent2_inheritance_history_file.txt', 'a') as parent2_inheritance_history_file:
            parent2_inheritance_history_file.write(str(winner.parent2_fit) + '\n')
        
        with open(save_dir + 'pure_sample_fitness_history_file.txt', 'a') as psfhf:    
            if -1 not in SELECTED: 
                inp = []
                out = []         
                for s in SELECTED:
                    inp.append( TR_INPUTS[s])
                    out.append(TR_OUTPUTS[s])
                fit = ef(winner, config, inp, out)
                psfhf.write(str(fit) + '\n')
            else:
                fit = ef(winner, config, TR_INPUTS, TR_OUTPUTS)
                psfhf.write(str(fit) + '\n')                      
        
        try:
            current_best_fit = ef(winner, config, TR_INPUTS, TR_OUTPUTS)
        except:
            current_best_fit = 0.0
        current_best_sel, ngenes  = eval_functions.features_selected(winner, config, 1.0)
        print('Training error: {} with {} genes ({})'.format(current_best_fit, ngenes, current_best_sel))
        #best_general_fitness = sort_genomes(p.population, config, tr_inputs, tr_outputs)[0][1]
        #print('Best general fitness: ' + str(best_general_fitness))
        #with open(save_dir + 'general_fitness_history.txt', 'a') as gfh_file:
        #    gfh_file.write(str(best_general_fitness) + '\n')
        with open(save_dir + 'sample_fit_history.txt', 'a') as sample_history_file:
            sample_history_file.write(str(winner.fitness) + '\n')
        with open(save_dir + 'train_fit_history.txt', 'a') as fitness_history_file:
            fitness_history_file.write(str(current_best_fit) + '\n')
        with open(save_dir + 'nselected_history.txt', 'a') as n_selected_history_file:
            n_selected_history_file.write(str(current_best_sel) + '\n')
        with open(save_dir + 'ngenes_history.txt', 'a') as ngenes_history_file:
            ngenes_history_file.write(str(ngenes) + '\n')
        '''with open(save_dir + 'true_n_selected_history.txt', 'a') as tnsh_file:
            tfs = eval_functions.features_selected(true_winner, config, 1)
            print('Number of features: ' + str(tfs))
            tnsh_file.write(str(tfs)+'\n')
        temp_kcm, temp_tracc = log_error.report(tr_inputs, tr_outputs, classes_labels, classifier=true_winner, config=config, is_NEAT=True, is_ensemble=False, error_file_path=save_dir+'g'+str(generation)+'training_error.txt', conf_matrix=False, dataset_label='TRAINING',should_print=False)
        with open(save_dir + 'train_acc_history.txt', 'a') as trah_file:
            trah_file.write(str(temp_tracc)+'\n')
        print('Training accuracy: ' + str(temp_tracc))'''
        if te_inputs is not None and te_outputs is not None:
            try:
                current_best_test = ef(winner, config, te_inputs, te_outputs)
            except:
                current_best_test = 0.0
            print(' Testing error: {}'.format(current_best_test))
            with open(save_dir + 'test_fit_history.txt', 'a') as test_fitness_history_file:
                test_fitness_history_file.write(str(current_best_test) + '\n')
            '''temp_kcm, temp_teacc = log_error.report(te_inputs, te_outputs, classes_labels, classifier=true_winner, config=config, is_NEAT=True, is_ensemble=False, error_file_path=save_dir+'g'+str(generation)+'testing_error.txt', conf_matrix=False, dataset_label='TESTING', should_print=False)
            with open(save_dir + 'test_acc_history.txt', 'a') as teah_file:
                teah_file.write(str(temp_teacc)+'\n')
            print('Testing accuracy: ' + str(temp_teacc))'''
            with open(save_dir + 'overfitting.txt', 'a') as overfitting_file:
                overfitting_file.write('{}\n'.format(abs(current_best_test - current_best_fit)))

    with open(save_dir+'selection_history.txt', 'w') as selection_history_file:
        for sh in selection_history:
            selection_history_file.write(str(sh) + '\n')

    #SELECTED = [-1]
    #pe = neat.ParallelEvaluator(num_cores, eval_genome)  
    #winner = p.run(pe.evaluate, num_gen)

    # Display the winning genome.   
    with open(save_dir+'genome.txt', 'w') as genome_file:
        print('\nBest genome:\n{!s}'.format(winner))
        genome_file.write('Best genome:\n{!s}'.format(winner))

    node_names = {-(k+1): v for k, v in enumerate(features_names)}
    node_names.update({k: v for k, v in enumerate(classes_labels)})
    with open(save_dir+'node_names.txt', 'w') as nnf:
        for i in node_names:
            nnf.write(str((i, node_names[i]))+'\n')
    with open(save_dir+'features_names.txt', 'w') as fnf:
        for i in features_names:
            fnf.write(str(i)+'\n')        

    visualize.save_selection_history(gene_selection_history, BEST_NN_GENES, pvalues, probs, features_names, save_dir+'gshistory.csv')
    visualize.plot_selection_history(gene_selection_history, save_dir+'gshistory.png')

    print_population(p.end_population, config, node_names, tr_inputs, tr_outputs, save_dir) 
    visualize.plot_stats(stats, ylog=False, view=False, filename=save_dir+'avg_fitness.svg')
    visualize.plot_species(stats, view=False, filename=save_dir+'speciation.svg')
    visualize.draw_net(config, winner, view=False, node_names=node_names, filename=save_dir+'net')
    visualize.draw_net(config, winner, view=False, node_names=node_names, show_disabled=False, filename=save_dir+'net2')
    visualize.draw_required_net(config, winner, view=False, node_names=node_names, filename=save_dir+'reqnet')
    genes, nodes = visualize.required_nodes(config, winner, node_names=node_names, filename=save_dir+'nodes')
    for g in genes:
        GENES.append(g)
    NUM_GENES.append(len(genes))    

    try:
        visualize.boxplot(tr_inputs, save_dir+'genes/box_seltrain', genes=genes, label=True, classes=tr_outputs, classes_names=classes_labels)
    except:
        pass
    if te_inputs is not None:
        try:
            visualize.boxplot(te_inputs, save_dir+'genes/box_seltest', genes=genes, label=True, classes=te_outputs, classes_names=classes_labels) 
        except:
            pass
        try:
            visualize.boxplot2(tr_inputs, te_inputs, tr_outputs, te_outputs, save_dir+'genes/box_selALL', genes=genes, label=True, classes_names=classes_labels)          
        except:
            pass

    try:
        visualize.plot_distribution(tr_inputs, save_dir+'genes/dist_seltrain', genes=genes, label=True)
    except:
        pass
    try:
        visualize.plot_distribution(tr_inputs, save_dir+'genes/dist_seltrain_cl', genes=genes, label=True, classes=tr_outputs, classes_names=classes_labels)    
    except:
        pass
    if te_inputs is not None:
        try:
            visualize.plot_distribution(te_inputs, save_dir+'genes/dist_seltest', genes=genes, label=True)
        except: 
            pass
        try:
            visualize.plot_distribution(te_inputs, save_dir+'genes/dist_seltest_cl', genes=genes, label=True, classes=te_outputs, classes_names=classes_labels)
        except:
            pass

    try: 
        visualize.plot_selected_genes_expression(tr_inputs, genes, tr_outputs, classes_labels, save_dir+'tr_sel_genes_exp')
    except:
        pass
    try:
        visualize.plot_genes_correlation(tr_inputs, genes, save_dir+'tr_genes_correlation')
    except:
        pass
    try:
        visualize.save_gene_selection(sel_gct_file, genes, save_dir+'tr_selection')
    except:
        pass
    '''voters = ensemble(p, config, 1, node_names, save_dir, tr_inputs, tr_outputs, classes_labels=classes_labels)
   
    for i in xrange(len(voters)+1):
        vs = voters[0:i+1]
        tr_kcm, tr_acc = log_error.report(tr_inputs, tr_outputs, classes_labels, classifier=vs, config=config, is_NEAT=True, is_ensemble=True, error_file_path=save_dir+'ensemble/'+str(i+1)+'tra_error.txt', conf_matrix=False, dataset_label='TRAINING', should_print=False)
        te_kcm, te_acc = log_error.report(te_inputs, te_outputs, classes_labels, classifier=vs, config=config, is_NEAT=True, is_ensemble=True, error_file_path=save_dir+'ensemble/'+str(i+1)+'tes_error.txt', conf_matrix=False, dataset_label='TESTING', should_print=False)
        with open(save_dir+'voters_tr_acc.txt', 'a') as vtrc_file:
            vtrc_file.write(str(tr_acc) + '\n')
        with open(save_dir+'voters_te_acc.txt', 'a') as vtec_file:
            vtec_file.write(str(te_acc) + '\n')'''
    
    ERROR_NAMES = []
    bla, bla2, error_names = log_error.report(tr_inputs, tr_outputs, classes_labels, names=tr_names, classifier=winner, config=config, is_NEAT=True, is_ensemble=False, error_file_path=save_dir+'training_error.txt', conf_matrix=True, dataset_label='TRAINING')
    for en in error_names:
        ERROR_NAMES.append(en)
    k_conf_matrix = None
    acc = 0.0
    if te_inputs is not None and te_outputs is not None:
        try:
            visualize.plot_selected_genes_expression(te_inputs, genes, te_outputs, classes_labels, save_dir+'te_sel_genes_exp')
        except:
            pass
        try:
            visualize.plot_genes_correlation(te_inputs, genes, save_dir+'te_genes_correlation')
        except:
            pass
        k_conf_matrix, acc, error_names = log_error.report(te_inputs, te_outputs, classes_labels, names=te_names, classifier=winner, config=config, is_NEAT=True, is_ensemble=False, error_file_path=save_dir+'testing_error.txt', conf_matrix=True, dataset_label='TESTING')    
        for en in error_names:
            ERROR_NAMES.append(en)    
    return k_conf_matrix, acc, ERROR_NAMES, len(genes)

'''def plot_global_genes_stats(genes, training, testing=None, filename=''):
    global tr_data_file
    cnt = Counter(genes)
    genes = [k for k, v in cnt.iteritems() if v > 1]
    print(genes)
    classes_labels = training.classes_labels
    tr_mean, tr_std = training.get_mean_std()
    tr_inputs  = training.get_dataset(sampleXfeature=True, standardized=True)
    tr_outputs = training.get_classes_one_hot() 
    visualize.plot_selected_genes_expression(tr_inputs, genes, tr_outputs, classes_labels, filename+'tr_expression')
    visualize.plot_genes_correlation(tr_inputs, genes, filename+'tr_correlation')
    visualize.save_gene_selection(tr_data_file, genes, filename+'tr_selection')
    if testing is not None:
        te_inputs  = testing.get_dataset(sampleXfeature=True, standardized=True, mean=tr_mean, std=tr_std)
        te_outputs = testing.get_classes_one_hot()
        visualize.plot_selected_genes_expression(te_inputs, genes, te_outputs, classes_labels, filename+'te_expression')
        visualize.plot_genes_correlation(te_inputs, genes, filename+'te_correlation') '''  
    
if __name__ == '__main__':

    #global variables
    TR_INPUTS  = None
    TR_OUTPUTS = None
    SELECTED  = []
    GENES      = []
    NUM_GENES  = []  
    decay = 0.8
    ef = eval_functions.binary_logloss

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    
    save_dir  = sys.argv[1] + '/'
    num_gen   = int(sys.argv[2])
    num_cores = int(sys.argv[3])
    k         = int(sys.argv[4])
    if k == 1:
        sys.exit('Error: k must be 0 or larger than 1')
    tr_data_file_sel = ''    
    tr_data_file  = sys.argv[5]
    tr_label_file = sys.argv[6]
    indexes_file  = sys.argv[7]
    try:
        te_data_file  = sys.argv[8]
        te_label_file = sys.argv[9]
        independent_test = True
        print('Independent test set used')
    except IndexError:
        independent_test = False        
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #copyfile(config_path, save_dir+'c-n')
    try:
        copyfile('NEAT.py', save_dir+'n.py')
    except:
        pass
    try:
        copyfile('script.sh', save_dir+'s.sh')
    except:
        pass
    try:
        copyfile('eval_functions.py', save_dir+'ef.py')
    except:
        pass
    
    #READ TRAINING DATASET
    training = MAR(tr_data_file, tr_label_file)
    if training.number_classes != 2:
        sys.exit('ERROR: Number of classes must be 2!')
    print(training.get_mean_std())
    print('Selected decay is 1 - {}'.format(decay))
    
    #READ PRECOMPUTED CROSS-VALIDATION INDEXES
    if indexes_file == 'None':
        pre_indexes = None
    else:
        pre_indexes = ML_frame.get_pre_folds(k, indexes_file)    
       
    if independent_test:
        testing = MAR(te_data_file, te_label_file)
        ML_frame.start(None, run, training, testing=testing, k=k, save_dir=save_dir, num_cores=num_cores, num_gen=num_gen, pre_indexes=pre_indexes)
        if k > 0:
            visualize.plot_genes_stats(GENES, k, filename=save_dir+'genes')
        visualize.plot_genes_selection(NUM_GENES, training.number_data_features, filename=save_dir+'genes_selected') 
        #plot_global_genes_stats(GENES, training, testing=testing, filename=save_dir+'GLOBAL_GENES_')
    else:    
        ML_frame.start(None, run, training, k=k, save_dir=save_dir, num_cores=num_cores, num_gen=num_gen, pre_indexes=pre_indexes)
        if k > 0:
            visualize.plot_genes_stats(GENES, k, filename=save_dir+'genes')
        visualize.plot_genes_selection(NUM_GENES, training.number_data_features, filename=save_dir+'genes_selected') 
        #plot_global_genes_stats(GENES, training, filename=save_dir+'GLOBAL_GENES_')
