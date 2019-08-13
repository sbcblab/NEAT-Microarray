#BRUNO IOCHINS GRISCI
#JUNE 2018

import sys

run_range = [int(sys.argv[1]), int(sys.argv[2])+1] 
folder = sys.argv[3]
f      = sys.argv[4]

got = []
for r in xrange(run_range[0], run_range[1]):
    try:
        with open('{}/{:0>2}/{}'.format(folder, r, f), 'r') as rf:
            for line in rf.readlines():
                if ' 3-FOLD CROSS VALIDATION average' in line:
                    got.append(float(line.split(":", 1)[1]))
                    break
            else:
                got.append(None)
    except:
        got.append('Empty')        

print(folder[0:folder.find('/')]+'<-c'+str(tuple(got))) 
