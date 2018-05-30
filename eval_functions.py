#bruno iochins grisci

import math
import neatB
import numpy as np
from neat.graphs import required_for_output
from neatB.activations import sigmoid_activation
from neatB.activations import mgauss_activation
from sklearn.metrics import mean_squared_error
import pandas as pd
import copy

def c_e(net_output, one_hot):
    n = len(one_hot)
    C = 0.0
    precision_error = 0.0000001 
    try:
        for i in xrange(n):
            if one_hot[i] == 0.0:
                if net_output[i] == 1.0:
                    # print('---') 
                    #print('WARNING: precision error in cross-entropy')
                    # print(one_hot)
                    #print(net_output)
                    #print('---')             
                    C += math.log(precision_error)
                else:    
                    C += math.log(1.0 - net_output[i])
            elif one_hot[i] == 1.0:
                if net_output[i] == 0.0:
                    #print('---') 
                    #print('WARNING: precision error in cross-entropy')
                    #print(one_hot)
                    #print(net_output)
                    #print('---')             
                    C += math.log(precision_error)               
                else:
                    C += math.log(net_output[i])
            else:
                print('WARNING: output not one-hot coded! ' + str(one_hot))
                C += one_hot[i] * math.log(net_output[i]) + (1.0 - one_hot[i]) * math.log(1.0 - net_output[i])
    except:
        C += -999999.99
        print('Look at the c-e error here')
    return C

def cross_entropy(genome, config, inputs, outputs):
    fit = 0.0
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        output = net.activate(i)
        softmax_output = neatB.math_util.softmax(output)
        try:
            fit += c_e(softmax_output, o)
        except:
            print('#####')
            print('Error in cross-entropy')
            print(o)
            print(output)
            print(softmax_output)
            print('#####')
            print('\nProblematic genome:\n{!s}'.format(genome))
            print('#####')
            fit += c_e(softmax_output, o)
    fit = (fit / len(inputs))
    return fit

def imbalanced_cross_entropy(genome, config, inputs, outputs):
    fits = [0.0] * len(outputs[0])
    classes_sizes = [0.0] * len(outputs[0])
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        output = net.activate(i)
        softmax_output = neatB.math_util.softmax(output)
        class_index = np.argmax(o)
        try:
            fits[class_index] += c_e(softmax_output, o)
            classes_sizes[class_index] += 1.0
        except:
            print('#####')
            print('Error in cross-entropy')
            print(o)
            print(output)
            print(softmax_output)
            print('#####')
            print('\nProblematic genome:\n{!s}'.format(genome))
            print('#####')
            fits[class_index] += c_e(softmax_output, o)
            classes_sizes[class_index] += 1.0
    
    fit = 0.0
    for c in zip(fits, classes_sizes):
        fit += (c[0] / c[1])    
    fit = fit/float(len(fits))
    return fit    

def logloss(genome, config, inputs, outputs):
    loss = 0.0
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        output = net.activate(i)
        softmax_output = neatB.math_util.softmax(output)
        for y, p in zip(o, softmax_output):
            loss += y * math.log(max(p, 1e-300))  
    return loss / len(inputs) 
    
def binary_logloss(genome, config, inputs, outputs):
    loss = [0.0, 0.0]
    classes_sizes = [0.0, 0.0]
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        if type(o) is not list:
            o = o.tolist()
        y = float(o.index(max(o)))
        output = net.activate(i)    
        p = (output[0])
        #p = mgauss_activation(output[0])
        loss[int(y)] += y * math.log(max(p, 1e-300)) + (1.0 - y) * math.log(1.0 - min(p, 1.0-1e-16))  
        classes_sizes[int(y)] += 1.0
    fit = 0.0
    for l, s in zip(loss, classes_sizes):
        fit += l / s
    return fit / 2.0 
    
def binary_accuracy(genome, config, inputs, outputs):
    acc = 0.0
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        if type(o) is not list:
            o = o.tolist()
        y = float(o.index(max(o)))
        output = net.activate(i)    
        p = (output[0])
        #p = mgauss_activation(output[0])
        if abs(p-y) < 0.5:
            acc += 1.0 
    return acc / len(inputs)             

def neat_fitness(genome, config, inputs, outputs):
    fit = 0.0
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        if type(o) is not list:
            o = o.tolist()
        y = float(o.index(max(o)))
        output = net.activate(i)    
        p = (output[0])
        #p = mgauss_activation(output[0])
        fit += abs(p - y) 
    return (len(inputs) - fit)**2    

def hinge(genome, config, inputs, outputs):
    #loss = [0.0] * len(outputs[0])
    loss = 0.0
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        if type(o) is not list:
            o = o.tolist()
        output = net.activate(i)
        #softmax_output = neatB.math_util.softmax(output)
        correct_index = o.index(max(o))
        rest = copy.deepcopy(output)
        #rest.pop(correct_index)
        #loss[correct_index] += max(0.0, 1.0 + max(rest) - output[correct_index])
        if correct_index == 0.0:
            correct_index = -1.0
        loss += max(0.0, 1.0 - correct_index * output[0])
        #l = np.array(loss).mean()
    return -loss / len(inputs) 
    
def L2_reg(genome, lamb, n):
    # L2 regularization is used to control the size of the weigths as form of combat overfitting
    # The satandard L2 reg does not use the term num_conn, but since the number 
    # of total connections in NEAT is variable it was added 
    num_conn = 0.0
    total_w2 = 0.0
    
    for kc in genome.connections:
        if genome.connections[kc].enabled == True:
            num_conn += 1.0
            total_w2 += genome.connections[kc].weight**2

    for kn in genome.nodes:
        num_conn += 1.0
        total_w2 += genome.nodes[kn].bias**2
        
    reg = (lamb/(2.0*n)) * (total_w2/num_conn) 
    return reg           

def mean_squared_error(net_output, one_hot):
    n = float(len(one_hot))
    e = 0.0
    for i in zip(net_output, one_hot):
        e += (i[1] - i[0])**2
    return e/n

def MSE(genome, config, inputs, outputs):
    e = 0.0
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        output = net.activate(i)
        softmax_output = neatB.math_util.softmax(output)
        e += mean_squared_error(softmax_output, o)
    return -e/float(len(inputs))    

def imbalanced_MSE(genome, config, inputs, outputs):
    fits = [0.0] * len(outputs[0])
    classes_sizes = [0.0] * len(outputs[0])
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        output = net.activate(i)
        softmax_output = neatB.math_util.softmax(output)
        class_index = np.argmax(o)
        fits[class_index] += mean_squared_error(softmax_output, o)
        classes_sizes[class_index] += 1.0
    fit = 0.0
    for c in zip(fits, classes_sizes):
        fit += (c[0] / c[1])
    return fit
    
def accuracy(genome, config, inputs, outputs):
    correct = 0.0
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        o = o.tolist()
        output = net.activate(i)
        softmax_output = neatB.math_util.softmax(output)
        if max(softmax_output) != min(softmax_output):
            if o.index(max(o)) == softmax_output.index(max(softmax_output)):
                correct += 1.0
    acc = correct / len(inputs)
    return acc

def reward(genome, config, inputs, outputs):
    r = 0.0
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        o = o.tolist()
        output = net.activate(i)
        softmax_output = neatB.math_util.softmax(output)
        if max(softmax_output) != min(softmax_output):
            if o.index(max(o)) == softmax_output.index(max(softmax_output)):
                r += max(softmax_output)
            else:
                r -= max(softmax_output)
        else:
            r -= max(softmax_output)            
    return r / len(inputs)

def imbalanced_accuracy(genome, config, inputs, outputs):
    fits = [0.0] * len(outputs[0])
    classes_sizes = [0.0] * len(outputs[0])
    net = neatB.nn.FeedForwardNetwork.create(genome, config)
    for i, o in zip(inputs, outputs):
        o = o.tolist()
        output = net.activate(i)
        softmax_output = neatB.math_util.softmax(output)
        class_index = np.argmax(o)
        if max(softmax_output) != min(softmax_output):
            if o.index(max(o)) == softmax_output.index(max(softmax_output)):
                fits[class_index] += 1.0
        classes_sizes[class_index] += 1.0
    acc = 0.0
    for c in zip(fits, classes_sizes):
        acc += (c[0] / c[1])
    return acc / float(len(classes_sizes))
    
def features_selected(genome, config, lamb):
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k) 
    number_features = len(inputs)      
    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
    connections = set()
    for k, v in zip(genome.connections.keys(), genome.connections.values()):
        if v.enabled:
            connections.add(k)
    used_nodes = set(required_for_output(inputs, outputs, connections))
    selected_inputs = set()
    for k in inputs:
        for n in used_nodes:
            if (k, n) in connections:
                selected_inputs.add(k)        
    return lamb * (float(len(selected_inputs)) / number_features), len(selected_inputs)  
    #return  len(selected_inputs)

def std(genome, config, tri, lamb):
    inputs          = set()
    outputs         = set()
    connections     = set()
    selected_inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
    for k in config.genome_config.output_keys:
        outputs.add(k)
    for k, v in zip(genome.connections.keys(), genome.connections.values()):
        if v.enabled:
            connections.add(k)    
    used_nodes = set(required_for_output(inputs, outputs, connections))
    for k in inputs:
        for n in used_nodes:
            if (k, n) in connections:
                selected_inputs.add(k)
    indexes = (np.array(list(selected_inputs)) * -1) - 1               
    
    signals = []
    for i in indexes:
        for sample in tri:
            signals.append(sample[i])        
                    
    return lamb * (1.0/np.array(signals).std())
    
def sample_correlation(genome, config, dataset, lamb):
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k) 
    number_features = len(inputs)      
    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
    connections = set()
    for k, v in zip(genome.connections.keys(), genome.connections.values()):
        if v.enabled:
            connections.add(k)
    used_nodes = set(required_for_output(inputs, outputs, connections))
    selected_inputs = set()
    for k in inputs:
        for n in used_nodes:
            if (k, n) in connections:
                selected_inputs.add(k)        
    
    if len(selected_inputs) > 1:
        indexes = [abs(i)-1 for i in selected_inputs]
        data = np.array(dataset)[:, indexes]
        d = pd.DataFrame(data=data.transpose())
        corr = np.tril(d.corr(method='spearman').abs().get_values(), -1).flatten()
        corr = corr[corr != 0.0]
        return lamb * np.log(corr.mean())
    else:
        return lamb * -1.0

def minmax_class_distance(genome, config, nc, dataset, lamb):
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k) 
    number_features = len(inputs)      
    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
    connections = set()
    for k, v in zip(genome.connections.keys(), genome.connections.values()):
        if v.enabled:
            connections.add(k)
    used_nodes = set(required_for_output(inputs, outputs, connections))
    selected_inputs = set()
    for k in inputs:
        for n in used_nodes:
            if (k, n) in connections:
                selected_inputs.add(k)        

    indexes = [abs(i)-1 for i in selected_inputs]
    data = np.array(dataset)[:,indexes]
    classes = np.split(data, nc)
    
    current_min_distance = 100000.0
    for c in xrange(nc):
        for s1 in classes[c]:
            for oc in xrange(c+1, nc):
                for s2 in classes[oc]:
                    dist = np.linalg.norm(s1-s2)
                    if dist < current_min_distance:
                        current_min_distance = dist
    if current_min_distance == 0.0:
        current_min_distance = 0.00000001
    
    current_max_distance = 0.0
    for c in xrange(nc):
        for s1 in xrange(len(classes[c])):
            for s2 in xrange(s1+1, len(classes[c])):
                dist = np.linalg.norm(s1-s2)
                if dist > current_max_distance:
                    current_max_distance = dist
    
    return lamb * ((1.0/current_min_distance) + current_max_distance)
    #return lamb * current_max_distance                                        
            
def sample_distance(genome, config, nc, dataset, lamb):
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k) 
    number_features = len(inputs)      
    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
    connections = set()
    for k, v in zip(genome.connections.keys(), genome.connections.values()):
        if v.enabled:
            connections.add(k)
    used_nodes = set(required_for_output(inputs, outputs, connections))
    selected_inputs = set()
    for k in inputs:
        for n in used_nodes:
            if (k, n) in connections:
                selected_inputs.add(k)        

    indexes = [abs(i)-1 for i in selected_inputs]
    data = np.array(dataset)[:,indexes]
    classes = np.split(data, nc)
    mean_point = []
    for c in classes:
        mean_point.append(np.mean(c,0))
    intra_dist = []
    inter_dist = []
    for i in xrange(nc):
        for s in classes[i]:
            intra_dist.append(np.linalg.norm(s-mean_point[i]))
    
    #for p in mean_point:
    #
    inter_dist.append(np.linalg.norm(mean_point[0]-mean_point[1]))    
    
    tra_d = np.array(intra_dist).mean()
    ter_d = np.array(inter_dist).mean()  

    return lamb * ((1.0/ter_d) + tra_d)                                              
