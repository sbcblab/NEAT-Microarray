import sys
import os
import ast
import numpy as np

start = 1
end = 235
genes = []
acc = []

for fold in xrange(start, end+1):
    input_file  = 'neat_outLOOCV/{}fold/training_error.txt'.format(fold)
    l = [line.rstrip('\n') for line in open(input_file)]
    for thing in l:
        if 'TRAINING ERROR:' in thing:
            i0 = thing.find(':')+1
            i1 = len(thing)
            t = thing[i0:i1]
            a = float(t)
            acc.append(1.0-a)

print(acc)
print('###############################################')
print('AVERAGE           : {0:f}').format(np.mean(np.array(acc)))
print('STANDARD DEVIATION: {0:f}').format(np.mean(np.std(acc)))
