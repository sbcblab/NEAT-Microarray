from collections import Counter
import sys
import os
import ast
import numpy as np
import networkx as nx

from microarray_reader import MAR

dataset_file = sys.argv[1]
labels_file = sys.argv[2]
ma = MAR(dataset_file, labels_file)

start = 1
end = 235
output_file = 'connections'

labels = {}
for label in ma.classes_labels:
    labels.update({label: []})

for fold in xrange(start, end+1):
    connections = []
    input_file  = 'neat_outLOOCV/{}fold/genome.txt'.format(fold)
    l = [line.rstrip('\n') for line in open(input_file)]
    for thing in l:
        if 'True' in thing:
            i0 = thing.find('key=') + 4
            i1 = thing.find('),') + 1
            t = thing[i0:i1]
            pair = ast.literal_eval(t)
            connections.append(pair)

    G = nx.DiGraph()
    for c in connections:
        G.add_edge(c[0], c[1])

    genes = []    
    outputs = []
    for c in connections:
        if c[0] < 0 and c[0] not in genes:
            genes.append(c[0]) 
        if c[1] >= 0 and c[1] < 10 and c[1] not in outputs:
            outputs.append(c[1])            
        
    for g in genes:
        for o in outputs:
            if nx.has_path(G, g, o):    
                #print(ma.dataset_symbols[(-g)-1], ma.classes_labels[o])
                labels[ma.classes_labels[o]].append(ma.dataset_symbols[(-g)-1])

with open(output_file+'.txt', 'w') as out_file:            
    for d in labels:
        counter = Counter(labels[d])
        print d
        print counter
        out_file.write(d + ' ########################\n')
        for k,v in counter.most_common():
            out_file.write( "{} {}\n".format(k,v) )        
