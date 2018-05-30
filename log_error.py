# BRUNO IOCHINS GRISCI

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import neatB
import neatB.math_util
from neatB.activations import sigmoid_activation
from neatB.activations import mgauss_activation
import con_matrix
import numpy as np
import seaborn as sns

def convert_to_class(entry, class_label):
    return class_label[entry.index(max(entry))]
    
def convert_to_one_hot(entry, num_classes):
    one_hot = [0.0]*num_classes
    one_hot[entry] = 1.0
    return one_hot   

def plot_neatvis(dataset, filename):
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20,40))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(h_neg=190, h_pos=100, l=50, center='dark', as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dataset, cmap=cmap, vmin=-3.0, vmax=3.0, center=0.0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(filename+'.svg')
    plt.clf()
    plt.close() 
    
def plot_softvis(dataset, filename):
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20,40))
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dataset, vmin=0.0, vmax=1.0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(filename+'.svg')
    plt.clf()
    plt.close()      

def report(inputs, outputs, classes_labels, names, classifier, config=None, is_NEAT=False, is_ensemble=False, error_file_path='error.txt', conf_matrix=True, dataset_label='', should_print=True):
    # Show output of the most fit result against training data.
    error_names = []
    with open(error_file_path, 'w') as error_file:
        if should_print:
            print('\nOutput:')
        error_file.write('Output:')
        error = 0.0
        true_label = []
        pred_label = []
        neat_vision = []
        soft_vision = []
        
        if is_NEAT:
            clf = neatB.nn.FeedForwardNetwork.create(classifier, config)
        else:
            clf = classifier                
        
        for i, o, n in zip(inputs, outputs, names):
            if type(o) is not list:
                o = o.tolist()
     
            y = float(o.index(max(o)))
            if is_NEAT:
                output = clf.activate(i)
                p = output[0]
                #p = mgauss_activation(output[0])
                softmax_output = [1.0-p, p] 
            else:
                output = clf.predict([i])
                p = output.tolist()[0]
                softmax_output = [1.0-p, p]                   
       
            message = ''
            if abs(y - p) < 0.5:
                message = 'O'
            else:
                message = 'X'
                error += 1.0
                error_names.append(n)
            
            str_softmax_output = ['{:.2f}'.format(x) for x in softmax_output]
            if should_print:
                print("{!r} expected output {!r}, got {!r}".format(message, o, str_softmax_output))
            error_file.write("\n{!r} expected output {!r}, got {!r}".format(message, o, str_softmax_output))
            true_label.append(convert_to_class(o, classes_labels))
            pred_label.append(convert_to_class(softmax_output, classes_labels))

            neat_vision.append(output)
            soft_vision.append(softmax_output)
        
        plot_neatvis(neat_vision, error_file_path.replace('.txt', 'neat_vis'))
        plot_softvis(soft_vision, error_file_path.replace('.txt', 'soft_vis'))
                
        error = error / len(inputs)
        if should_print:
            print(dataset_label + " ERROR: " + str(error))
        error_file.write("\n" + dataset_label + " ERROR: " + str(error))

        if should_print:
            print("TRUE " + dataset_label + ": " + str(true_label))
            print("PRED " + dataset_label + ": " + str(pred_label))
        error_file.write("\nTRUE " + dataset_label + ": " + str(true_label))
        error_file.write("\nPRED " + dataset_label + ": " + str(pred_label))
        if should_print:
            print("########################################################")
            print("########################################################")
        error_file.write("\n########################################################")
        error_file.write("\n########################################################")     
        
        with open(error_file_path.replace('.txt', 'names.txt'), 'a') as error_name_file:
            for name in error_names:
                error_name_file.write(name+'\n') 
        
        if conf_matrix:
            return con_matrix.build(true_label, pred_label, classes_labels, error_file_path.replace('.txt', '')), 1.0 - error, error_names   
        else:
            return None, 1.0 - error, error_names

    
def report2(inputs, outputs, classes_labels, names, classifier, config=None, is_NEAT=False, is_ensemble=False, error_file_path='error.txt', conf_matrix=True, dataset_label='', should_print=True):  
    # Show output of the most fit result against training data.
    error_names = []
    with open(error_file_path, 'w') as error_file:
        if should_print:
            print('\nOutput:')
        error_file.write('Output:')
        error = 0.0
        true_label = []
        pred_label = []
        
        if is_NEAT and is_ensemble:
            print("is ENSEMBLE")
            clf = None
            voters = []
            for voter in classifier:
                voters.append(neatB.nn.FeedForwardNetwork.create(voter, config))
        elif is_NEAT and not is_ensemble:
            print("is NEAT")
            clf = neatB.nn.FeedForwardNetwork.create(classifier, config)
        else:
            print("is ELSE")
            clf = classifier
        
        for i, o, n in zip(inputs, outputs, names):
            if type(o) is not list:
                o = o.tolist()
            if is_NEAT and is_ensemble:
                outputs = []
                for voter in voters:
                    output = neatB.math_util.softmax(voter.activate(i))
                    '''if max(output) == min(output):
                        for j in xrange(len(output)):
                            output[j] = 0.0
                    else:
                        m = output.index(max(output))
                        for j in xrange(len(output)):
                            if j == m:
                                output[j] = 1.0
                            else:
                                output[j] = 0.0'''
                    outputs.append(output)
                softmax_output = np.array(outputs).mean(0).tolist() 
                #print(outputs)     
                #print(softmax_output)      
            elif is_NEAT and not is_ensemble:
                output = clf.activate(i)
                softmax_output = neatB.math_util.softmax(output)
            else:
                output = clf.predict([i])
                softmax_output = output.tolist()[0]        
            message = ''
            if type(softmax_output) is int:
                softmax_output = convert_to_one_hot(softmax_output, len(classes_labels))
            if max(softmax_output) == min(softmax_output):
                message = '?'
                error += 1.0
                error_names.append(n)
            elif o.index(max(o)) == softmax_output.index(max(softmax_output)):
                message = 'O'
            else:
                message = 'X'
                error += 1.0
                error_names.append(n)
            if should_print:
                print("{!r} expected output {!r}, got {!r}".format(message, o, softmax_output))
            error_file.write("\n{!r} expected output {!r}, got {!r}".format(message, o, softmax_output))
            true_label.append(convert_to_class(o, classes_labels))
            pred_label.append(convert_to_class(softmax_output, classes_labels))

        if is_NEAT and not is_ensemble:
            neat_vision = []
            soft_vision = []
            for i, o in zip(inputs, outputs):
                out = clf.activate(i)
                print(out)
                neat_vision.append(out)
                soft_vision.append(neatB.math_util.softmax(out))
            plot_neatvis(neat_vision, error_file_path.replace('.txt', 'neat_vis'))
            plot_softvis(soft_vision, error_file_path.replace('.txt', 'soft_vis'))
                
        error = error / len(inputs)
        if should_print:
            print(dataset_label + " ERROR: " + str(error))
        error_file.write("\n" + dataset_label + " ERROR: " + str(error))

        if should_print:
            print("TRUE " + dataset_label + ": " + str(true_label))
            print("PRED " + dataset_label + ": " + str(pred_label))
        error_file.write("\nTRUE " + dataset_label + ": " + str(true_label))
        error_file.write("\nPRED " + dataset_label + ": " + str(pred_label))
        if should_print:
            print("########################################################")
            print("########################################################")
        error_file.write("\n########################################################")
        error_file.write("\n########################################################")     
        
        with open(error_file_path.replace('.txt', 'names.txt'), 'a') as error_name_file:
            for name in error_names:
                error_name_file.write(name+'\n') 
        
        if conf_matrix:
            return con_matrix.build(true_label, pred_label, classes_labels, error_file_path.replace('.txt', '')), 1.0 - error, error_names   
        else:
            return None, 1.0 - error, error_names
