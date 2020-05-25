fname = 'socialnetworkfile.txt'
pl_fname = 'peerleadersfile.txt'

g = load_edgelist(fname)
pl_list = strip(pl_fname)


import networkx as nx

total = 0
overlap = []
pl_list2  = pl_list[1:]
total_degree = 0

for node in pl_list:
    overlap = [x for x in g.neighbors(node) if x in pl_list2]
    overlap = len(overlap)
    print(overlap)
    skip = [x for x in pl_list if x not in pl_list2]
    print(skip)
    degree = len([x for x in g.neighbors(node) if x not in skip])
    print(degree)
    total_degree = total_degree + degree
    total = total + overlap
    if pl_list2: pl_list2.pop(0)

    
print(total/float(total_degree))