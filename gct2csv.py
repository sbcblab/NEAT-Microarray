#Bruno Iochins Grisci
#DECEMBER/2017

import sys
import numpy as np

from microarray_reader import MAR

if __name__ == '__main__':
    dataset_file = sys.argv[1]
    labels_file = sys.argv[2]
    
    ma = MAR(dataset_file, labels_file)
    dataset = ma.get_dataset(sampleXfeature=True, standardized=True)
    features_names = ma.dataset_symbols
    classes = ma.get_classes()
    
    if (0 in classes):
        classes = map(lambda x:x+1, classes)
    
    with open(dataset_file.replace('.gct', '.csv'), 'w') as csv_file:
        header = 'class'
        for name in features_names:
            header = header + ', ' + name
        header = header + '\n'
        csv_file.write(header)
        for d in zip(classes, dataset):
            line = str(d[0])
            for v in d[1]:
                line = line + ', ' + str(v)
            line = line + '\n'
            csv_file.write(line)            


