#Bruno Iochins Grisci
#MARCH/2018

import sys
import csv
import numpy as np
from collections import OrderedDict

if __name__ == '__main__':
    tab_file = sys.argv[1]
    
    gct_file = tab_file.replace('.tab', '_tab.gct')
    cls_file = tab_file.replace('.tab', '_tab.cls')
  
    genes   = []
    classes = []
    with open(tab_file,'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            if '' not in line:
                if line[0] == 'gene' or line[0] == 'discrete' or line[0] == 'class':
                    pass
                else:
                    classes.append(line[0])
                genes.append(line[1:])
        
    genes = np.array(genes)
    genes = genes.transpose()
    print(genes)  
    
    n_genes   = genes.shape[0]
    n_samples = genes.shape[1] - 2 
    labels = list(OrderedDict.fromkeys(classes))
    n_classes = len(labels)
    class_index = []
    for lab in classes:
        class_index.append(str(labels.index(lab))) 
    print(classes)
    print(class_index)
    print(labels)
    print(n_genes, n_samples, n_classes)
    
    if len(classes) != n_samples:
        print('ERROR') 
        
    description = ''
    for i in xrange(len(classes)):
        description = description + '\t' + classes[i] + str(i+1)
    np.savetxt(gct_file, genes, fmt='%s', delimiter='\t', newline='\n', header='#0.0\n{} {}\nName\tDescription'.format(n_genes, n_samples) + description, comments='') 
        
    with open(cls_file, 'w') as f:
        f.write('{} {} {}\n'.format(n_samples, n_classes, 1))
        f.write('# ' + ' '.join(labels) + '\n')
        f.write(' '.join(class_index))
        
                   
                
