#Bruno Iochins Grisci
#SEPTEMBER/2017
#This code gets the data from a microarray file

import copy
import numpy as np
import math
import shlex
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut

class MAR:

    def __init__(self, data_file_name, class_file_name=None):
        self.data_file_name = data_file_name
        self.class_file_name = class_file_name
        
        self.data_comments = ""
        self.number_data_features = 0
        self.number_data_samples = 0
        self.dataset_header = []
        self.dataset_symbols = []
        self.dataset_descriptions = []
        self.dataset_complete = []
        self.dataset_featureXsample = []
        
        self.number_labels_samples = 0
        self.number_classes = 0
        self.other_labels_info = 0
        self.classes_labels = []
        self.classes = []
        
        self.read_dataset()
        self.read_labels()
        self.validate_samples_sizes()    
        
    def read_dataset(self):
        with open(self.data_file_name, 'r') as data_file: 
            for i, line in enumerate(data_file):
                if i == 0:
                    self.data_comments = line
                elif i == 1:
                    info_line = map(int, line.split())
                    self.number_data_features = info_line[0]
                    self.number_data_samples = info_line[1]
                elif i == 2:
                    self.dataset_header = line.split()               
                elif i > 2:
                    row = shlex.split(line)
                    self.dataset_symbols.append(row[0])
                    self.dataset_descriptions.append(row[1])
                    self.dataset_complete.append(row)
                    self.dataset_featureXsample.append(map(float, row[2:]))            
        self.dataset_featureXsample = np.array(self.dataset_featureXsample)            
                               
    def read_labels(self):
        if self.class_file_name is not None:
            with open(self.class_file_name, 'r') as class_file: 
                for i, line in enumerate(class_file):
                    if i == 0:
                        info_line = map(int, line.split())
                        self.number_labels_samples = info_line[0]
                        self.number_classes        = info_line[1]
                        self.other_labels_info     = info_line[2]
                    elif i == 1:
                        self.classes_labels = line.split()[1:]
                    elif i == 2:
                        self.classes = map(int, line.split())
                        break
                        
    def validate_samples_sizes(self):
        if self.number_data_samples != self.number_labels_samples:
            print('WARNING: Different number of samples in each file.')                 
        if self.number_data_samples != len(self.dataset_header) - 2:
            print('WARNING: Different number of samples in info and dataset.')   

        sizes = []
        for i in self.dataset_featureXsample:
            sizes.append(len(i))
        my_dict = {i:sizes.count(i) for i in sizes}   
        if len(my_dict) > 1:
            print(my_dict)
            print('WARNING: Missing or extra item in the dataset.')

    def get_dataset_sampleXfeature(self):
        return np.transpose(self.dataset_featureXsample)
    
    def rescale_dataset(self, dataset, min_value, max_value):
        rds = []
        for row in dataset:
            a = min(row)
            b = max(row)
            nrow = map(lambda x: ((max_value-min_value)/(b-a))*(x-a)+min_value, row)
            rds.append(nrow)
        return np.array(rds)                                           

    #@staticmethod
    #def scale_dataset(dataset, min_values, max_values):
    #(1.0 / max_values - min_values) * (dataset - min_values)

    @staticmethod
    def standardize_dataset(dataset, ds_mean, ds_std):
        if ds_mean is None:
            ds_mean = np.mean(dataset, axis=0)
        if ds_std is None:
            ds_std  = np.std(dataset, axis=0)
        ds_std[ds_std == 0.0] = 0.00001    
        nds = (dataset - ds_mean) / ds_std
        return nds 

    def get_mean_std(self):
        ds = self.get_dataset_sampleXfeature()
        return np.mean(ds, axis=0), np.std(ds, axis=0)            
    
    def get_dataset(self, sampleXfeature=True, rescaled=False, min_value=0.0, max_value=1.0, standardized=False, mean=None, std=None):
        ds = None
        if sampleXfeature:
            ds = self.get_dataset_sampleXfeature()
        else:
            ds = self.dataset_featureXsample              
        if rescaled:
            return self.rescale_dataset(ds, min_value, max_value)
        elif standardized:
            return self.standardize_dataset(ds, mean, std)
        else:
            return ds                 

    def get_classes(self):
        return self.classes
        
    def get_classes_one_hot(self):
        one_hot = []   
        ranks = copy.deepcopy(self.classes)
        while min(ranks) > 0:
            ranks = map(lambda x:x-1, ranks)      
        for sample in ranks:
            oh = [0.0]*self.number_classes
            oh[sample] = 1.0
            one_hot.append(oh)
        return one_hot
        
    def get_k_folds(self, k, sampleXfeature=True, standardized=True, one_hot=True, save_dir='', pre_indexes=None):
        num_samples = self.number_data_samples
        x = self.get_dataset(sampleXfeature=sampleXfeature)
        y = self.get_classes()
        yoh = self.get_classes_one_hot()

        skf = StratifiedKFold(n_splits=k, shuffle=True)
        skf.get_n_splits(x, y)       
        #loo = LeaveOneOut()
        #loo.get_n_splits(x, y)
        #print(loo)
        
        trt = 0
        tet = 0
        trx_folds = []
        try_folds = []
        tr_names = []
        tex_folds = []
        tey_folds = []
        te_names = []     
        
        sample_names = np.array(self.dataset_header[2:])
        
        if pre_indexes is None:
            pre_indexes = skf.split(x, y)   
        with open(save_dir+'cv_indexes.csv', 'w') as indexes_file:
            for train_index, test_index in pre_indexes:
            #for train_index, test_index in loo.split(x, y):
                print('---')
                if standardized:
                    tr_ds = np.take(x, train_index, axis=0)
                    te_ds = np.take(x, test_index,  axis=0)
                    tr_mean = np.mean(tr_ds, axis=0)
                    tr_std  = np.std(tr_ds, axis=0)
                    print('Fold tr len,  Fold te len: ', len(tr_ds), len(te_ds))
                    print('Fold tr mean, Fold te mean:', tr_mean, np.mean(te_ds, axis=0))
                    print('Fold tr std,  Fold te std: ', tr_std, np.std(te_ds, axis=0))
                    trx_folds.append(self.standardize_dataset(tr_ds, tr_mean, tr_std))
                    tex_folds.append(self.standardize_dataset(te_ds, tr_mean, tr_std))
                    tr_names.append(np.take(sample_names, train_index))
                    te_names.append(np.take(sample_names, test_index))
                else:
                    trx_folds.append(np.take(x, train_index, axis=0))
                    tex_folds.append(np.take(x, test_index, axis=0))
                    tr_names.append(np.take(sample_names, train_index))
                    te_names.append(np.take(sample_names, test_index))                              
                if one_hot:
                    try_folds.append(np.take(yoh, train_index, axis=0))
                    tey_folds.append(np.take(yoh, test_index, axis=0))               
                else:
                    try_folds.append(np.take(y, train_index, axis=0))
                    tey_folds.append(np.take(y, test_index, axis=0))
                print("TRAIN:", train_index, "TEST:", test_index)
                print("TRAIN:", len(train_index), "TEST:", len(test_index))
                train_line = 'TR'
                for tri in train_index:
                    train_line = train_line + ', ' + str(tri)
                train_line = train_line + '\n'                    
                indexes_file.write(train_line)   
                test_line = 'TE'
                for tei in test_index:
                    test_line = test_line + ', ' + str(tei)
                test_line = test_line + '\n'                    
                indexes_file.write(test_line)                              
                trt += len(train_index)
                tet += len(test_index)                
        
        print("TRAIN:", trt, "TEST:", tet)        
        return trx_folds, try_folds, tex_folds, tey_folds, tr_names, te_names
                           
