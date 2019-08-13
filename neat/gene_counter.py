from collections import Counter
import sys
import os
import ast
import numpy as np

from microarray_reader import MAR

dataset_file = sys.argv[1]
labels_file = sys.argv[2]

ma = MAR(dataset_file, labels_file)

start = 1
end = 235
output_file = 'LOOCV_gene_count_wod'
genes = []
n_genes = []

for fold in xrange(start, end+1):
    input_file  = 'neat_outLOOCV/{}fold/nodes_genes.txt'.format(fold)
    l = [line.rstrip('\n') for line in open(input_file)]
    genes = genes + l
    n_genes.append(len(l))

counter = Counter(genes)

print('###############################################')
print('GENES FREQUENCY')
with open(output_file+'.txt', 'w') as gene_file:
    s = sorted(counter, key=lambda x: counter[x], reverse=True)
    for k in s:
        index = -ast.literal_eval(k)[0]
        print('[{:3d}] {:22s}: {}'.format(int(counter[k]), k, int(counter[k]) * '*'))
        print(ma.dataset_descriptions[index-1])
        gene_file.write('[{:3d}] {:22s}: {}\n'.format(int(counter[k]), k, int(counter[k]) * '*'))
        #gene_file.write(ma.dataset_descriptions[index-1]+'\n')
    print('AVERAGE NUMBER OF GENES: {}').format(np.mean(np.array(n_genes)))
    print('STANDARD DEVIATION: {}').format(np.mean(np.std(n_genes)))
