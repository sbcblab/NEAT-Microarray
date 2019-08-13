#BRUNO IOCHINS GRISCI
#JULY 2018

import sys

run_range = [int(sys.argv[1]), int(sys.argv[2])+1]
folder = sys.argv[3]
f      = sys.argv[4]

total = 3*(run_range[1] - run_range[0] + 1)

c = {}
for r in xrange(run_range[0], run_range[1]):
    try:
        with open('{}/{:0>2}/{}'.format(folder, r, f), 'r') as rf:
            for line in rf.readlines():
                if '[' in line:
                    n = int(line[line.find('[')+1 : line.find(']')])
                    gene =  line[line.find(']')+1 : line.find(':')].replace(' ', '')
                    if gene in c:
                        c[gene] = c[gene] + n
                    else:
                        c[gene] = n    
    except:
        if 'Empty' in c:
            c['Empty'] = c['Empty'] + 1
        else:
            c['Empty'] = 1

#print(c)
t = 0
for k in c:
    t += c[k]  

i = 0
g05 = 0
for k, v in sorted(c.items(), key=lambda p:p[1], reverse=True):
    if i < 10:
        print(k, v, float(v)/float(total))
    if float(v)/float(total) >= 0.05:
        g05 += 1
    i += 1

print(len(c), t) 
print(g05) 
