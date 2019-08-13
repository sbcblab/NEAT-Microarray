#BRUNO IOCHINS GRISCI
#JUNE 2018

import sys

run_range = [int(sys.argv[1]), int(sys.argv[2])+1] 
fo1 = sys.argv[3]
fo2 = sys.argv[4]
fo3 = sys.argv[5]
fo4 = sys.argv[6]
gse = sys.argv[7]
f   = sys.argv[8]

for fo in [fo1, fo2, fo3, fo4]:
    got = []
    folder = fo + gse
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

print("count <- list({}, {}, {}, {})".format(fo1[0:-1], fo2[0:-1], fo3[0:-1], fo4[0:-1]))
print("posthoc.kruskal.dunn.test(x=count, p.adjust.method=\"bonferroni\")") 
