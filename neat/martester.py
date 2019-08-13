#Bruno Iochins Grisci
#SEPTEMBER/2017

import sys
import numpy as np

from microarray_reader import MAR

dataset_file = sys.argv[1]
labels_file = sys.argv[2]

ma = MAR(dataset_file, labels_file)

print("{!r}, {!r}, {!r}".format(ma.data_comments, ma.number_data_features, ma.number_data_samples))
print(ma.dataset_header)
print("{!r} = {!r}".format(len(ma.dataset_symbols), len(ma.dataset_descriptions)))
#print(ma.dataset_complete)

print('###')

dfs = ma.get_dataset(sampleXfeature=False)
print(dfs)
print(type(dfs))
print("{!r} x {!r}".format(len(dfs), len(dfs[0])))
print(dfs.shape)
m, s = ma.get_mean_std()
print(m)
print(m.shape)
print(s)
print(s.shape)
print(ma.get_dataset(sampleXfeature=False, rescaled=True, min_value=0, max_value=1))
print('---')
print(ma.get_dataset(sampleXfeature=False, standardized=True))

print('###')

dsf = ma.get_dataset(sampleXfeature=True)
print(dsf)
print(type(dsf))
print("{!r} x {!r}".format(len(dsf), len(dsf[0])))
print(dsf.shape)
m, s = ma.get_mean_std()
print(m)
print(m.shape)
print(s)
print(s.shape)
print(ma.get_dataset(rescaled=True, min_value=0, max_value=1))
print('---')
print(ma.get_dataset(standardized=True))

print("###")
print("{!r}, {!r}, {!r}".format(ma.number_labels_samples, ma.number_classes, ma.other_labels_info))
print(ma.classes_labels)
print(ma.get_classes())
print(ma.get_classes_one_hot())

#trx_folds, try_folds, tex_folds, tey_folds = ma.get_k_folds(10)
#for trx, tryy, tex, tey in zip(trx_folds, try_folds, tex_folds, tey_folds):
#    print(trx, tryy, tex, tey)
