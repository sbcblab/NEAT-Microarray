#Bruno Iochins Grisci
#MAY/2018

import sys
import csv
import numpy as np

if __name__ == '__main__':
    csv_file = sys.argv[1]
    gct_file = csv_file.replace('.csv', '.gct')
    cls_file = csv_file.replace('.csv', '.cls')
    
    data = np.genfromtxt(csv_file, delimiter=',', dtype=None)
    for i in xrange(data[0].size):    
        if data[0][i][1:3] == 'x.':
            data[0][i] = '"' + data[0][i][3:]
    data = np.insert(data, 0, data[0], axis=0)
    data= data.transpose()

    classes = data[-1]
    data = data[:-1]
    data[0] = np.core.defchararray.add(data[0], classes)
    data[0][0] = 'symbol'
    data[0][1] = 'symbol'
    for i in xrange(data[0].size):
        data[0][i] = data[0][i].replace('"','')
    
    header = (data.shape[0] - 1, data.shape[1] - 2)
    dheader = '#0.0\n{} {}'.format(header[0], header[1])
    print(dheader)
    print(data)
    np.savetxt(gct_file, data, delimiter=' ', header=dheader, comments='', fmt='%s')
    
    classes = classes[2:]
    names = np.unique(classes)
    all_names = '#'
    for n in names:
        all_names = all_names + ' ' + n
    all_names = all_names.replace('"', '')
    cheader = "{} {} 1\n{}".format(classes.shape[0], names.shape[0], all_names)
    print(cheader)
    
    labels = []
    for label in classes:
        labels.append(np.where(names==label)[0][0])
    labels = np.array(labels)
    print(labels)
    np.savetxt(cls_file, labels.reshape(1, labels.shape[0]), delimiter=' ', header=cheader, comments='', fmt='%s')
    
