#Bruno Iochins Grisci
#July 2018

import math
import numpy as np

def bin(A, a):
    return float(math.factorial(A)) / float((math.factorial(a) * math.factorial(A-a)))

def Pg(m, G, a, k=3, r=31):
    p = float(m)/float(G)
    A = r*k
    return bin(A, a) * (p**a) * ((1-p)**(A-a))

def Pgreat(m, G, a):
    t = []
    i = 0
    while i < a:
        t.append(Pg(m, G, i))
        i += 1
    t = np.array(t)
    print(t)
    print(t.sum())
    return '{:0.2e}'.format(1.0 - t.sum())

