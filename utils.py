def load(fname, directed):
    import numpy as np
    a = np.loadtxt(fname, dtype=np.int)
    a[:,:2] -= 1    #convert nodes to 0-indexing
    numnodes = len(np.unique(a))
    g = np.zeros((numnodes, numnodes))
    for i in range(a.shape[0]):
        g[a[i, 0], a[i, 1]] = 1
        if not directed:
            g[a[i, 1], a[i, 0]] = 1
    return g

def strip(fname):
    import networkx as nx
    with open(fname) as f:
        lines = f.readlines()
    lines = [int(x.strip()) for x in lines]
    return lines

def strip_demo(fname):
    import networkx as nx
    with open(fname) as f:
        lines = [line.split() for line in f]

    new_list = []
    for sublist in lines:
        new_sublist = []
        for each in sublist:
            each = int(each)
            new_sublist.append(each)
        new_list.append(new_sublist)

    return new_list

def load_edgelist(fname):
    import numpy as np
    import networkx as nx
    a = np.loadtxt(fname, dtype=np.int)
    #make_ids_contiguous(a)
    g = nx.from_edgelist(a)
    return g
    
def make_ids_contiguous(a):
    '''
    Input: the array from the original text file input
    Output: the same array, but with the node IDs in a contiguous block starting
    from 0. This lets the node IDs serve as array indices.
    '''
    import numpy as np
    next_id = 0
    id_map = {}
    for u in np.unique(a[:,:2]):
        id_map[u] = next_id
        next_id += 1
        a[:,:2][a[:,:2] == u] = id_map[u]
    return id_map

def load_netscience():
    import numpy as np
    import networkx as nx
    a = np.loadtxt('netscience.net', dtype=np.int)
    a = a[:, :2]
    make_ids_contiguous(a)
    return nx.from_edgelist(a)

def load_g(netname):
    import networkx as nx
    import numpy as np
    if netname == 'netscience':
        g = load_netscience()
    elif netname == 'homeless-a':
        G = load('a1.txt', directed=False)
        g = nx.from_numpy_matrix(G)
    elif netname == 'homeless-b':
        G = load('b1.txt', directed=False)
        g = nx.from_numpy_matrix(G)
    elif 'india' in netname:
        num = netname.split('-')[1]
        G = np.loadtxt('relations/' + num + '-All2.csv', delimiter=',')
        g = nx.from_numpy_matrix(G)
    elif netname == 'genrel':
        g = load_edgelist('genrel.txt')
    elif netname == 'SBM-1500-1':
        g = nx.read_edgelist('SBM_1500_1_1', nodetype=int)
    elif netname == 'SBM-equal_1000_0':
        g = nx.read_adjlist('SBM-equal_1000_0', nodetype=int)
    elif netname == 'SBM-unequal_1000_0':
        g = nx.read_adjlist('SBM-unequal_1000_0', nodetype=int)
    elif netname == 'residence':
        g = nx.read_edgelist('residence.txt', nodetype=int)
    return g

"""
Draw degree histogram with matplotlib.
Random graph shown as inset
"""
def degree_distribution(fname, list_of_nodes, isolates):
    import collections
    import matplotlib.pyplot as plt
    import networkx as nx

    G = load_edgelist(fname)
    r = {}
    G.add_nodes_from(isolates)

    for x in list_of_nodes:
        r[x] = G.degree()[x]

    degree_sequence=sorted([d for d in r.values()], reverse=True) # degree sequence

    degreeCount=collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    # ax.set_xticks([d+0.4 for d in deg])
    # ax.set_xticklabels(deg)

    # # draw graph in inset
    # plt.axes([0.4, 0.4, 0.5, 0.5])
    # Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
    # pos=nx.spring_layout(G)
    # plt.axis('off')
    # nx.draw_networkx_nodes(G, pos, node_size=20)
    # nx.draw_networkx_edges(G, pos, alpha=0.4)

    # plt.savefig("degree_histogram.png")
    plt.show()

# how many PLs are they connected to on average? 
def degree_withPL(fname, list_of_interest, pl_fname, isolates):
    import collections
    import matplotlib.pyplot as plt
    import networkx as nx

    G = load_edgelist(fname)
    r = {}
    G.add_nodes_from(isolates)

    pl_list = strip(pl_fname)

    for node in list_of_interest:
        a = [x for x in G.neighbors(node) if x in pl_list]
        r[node] = len(a)

    degree_sequence=sorted([d for d in r.values()], reverse=True) # degree sequence

    degreeCount=collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    #print(degree_sequence)
    plt.show()

#makes dictionary of shortest path lengths keyed by PLs
#each value is itself another dict, keyed by node number with value path length
def social_distance(g, list_of_interest):
    import networkx as nx
    d={}

    for node in list_of_interest:
        d[node] = nx.shortest_path_length(g,source=node)
    return d

def distance_from_PL(fname, pl_fname, convert_fname, nc_fname, isolates_fname):
    import networkx as nx
    import collections
    import matplotlib.pyplot as plt

    g = load_edgelist(fname)
    pl = strip(pl_fname)
    convert = strip(convert_fname)
    nc = strip(nc_fname)
    isolates = strip(isolates_fname)
    behavioral = convert + nc
    behavioral_without = [x for x in behavioral if x not in isolates]
    r = {} #large dictionary, keys are behavioral
    print(len(isolates), "length")
    d = social_distance(g, behavioral_without)
    for node in behavioral:
        if node in isolates:
            r[node] = 0
        else:
            p = {} #within dictionary, keys are pl, vals are distance, only take minimum in the end
            for peer_leader in pl:
                try:
                    p[peer_leader] = d[node][peer_leader]
                except KeyError:
                    print(peer_leader, node)
                    p[peer_leader]= 5
            min_distance = min(p.values())
            r[node] = min_distance
    print(r)

    for node in r.keys():
        if r[node]==5:
            r[node]=0
    c={}
    for each in convert:
        c[each] = r[each]

    distance_convert = sorted([a for a in c.values()], reverse=True)
    distance_sequence = sorted([a for a in r.values()], reverse=True) # degree sequence

    distanceCount=collections.Counter(distance_sequence)
    distanceConvert = collections.Counter(distance_convert)
    dis_c, cnt_c = zip(*distanceConvert.items())
    dis, cnt = zip(*distanceCount.items())
    print(distance_convert, len(distance_convert))
    print(distance_sequence, len(distance_sequence))

    plt.figure(1)
    plt.subplot(211)
    plt.bar(dis, cnt, width=0.80, color='b')

    plt.title("Distance from PL Histogram")
    plt.ylabel("Count")
    plt.xlabel("Distance")

    plt.subplot(212)
    plt.bar(dis_c, cnt_c, width=0.80, color = 'r')

    plt.ylabel("Count")
    plt.xlabel("Distance")
    # plt.show()


def visualize_set_separate(ICM, p, g, S, converted_list, total, fname, pl_fname, convert_fname, nc_fname, all_nodes):
    import networkx as nx
    import matplotlib.pyplot as plt

    pos=nx.spring_layout(g)
    total = [x for x in total if x not in S]
    converts = [x for x in converted_list if x in total]
    converts = [x for x in converts if x not in S]
    missing_converts = [x for x in converted_list if x not in total]
    missing_converts = [x for x in missing_converts if x not in S]
    
    n = [x for x in g.nodes() if x not in converted_list]
    missing_n = [x for x in n if x not in total]
    still_n = [x for x in n if x in total]

    all_n = still_n + missing_n
    n_colors = ['#565757']*len(still_n) + ['#E0E4E5']*len(missing_n)
    all_converts = converts + missing_converts
    converts_colors = ['#00C8FF']*len(converts) + ['#C1F0FE']*len(missing_converts)
  
    print('converts',all_converts)
    print('not', all_n)

    plt.figure(1)
    plt.subplot(211)
    nx.draw_networkx_labels(g, pos=pos,with_labels=True,font_size=8)

    nx.draw_networkx_nodes(g, pos= pos, nodelist= S, node_color = 'b', node_size=300)
    nx.draw_networkx_nodes(g, pos=pos, nodelist=all_converts, node_color = converts_colors, node_size=200)
    nx.draw_networkx_nodes(g, pos=pos, nodelist=all_n, node_color = n_colors, node_size = 200)

    # edges with missing nodes
    missing_edges = g.edges(missing_n + missing_converts)
    edge_colors = ['#E3E3E3'] * len(missing_edges)
    nx.draw_networkx_edges(g, pos=pos, width = 1.0)
    nx.draw_networkx_edges(g, pos=pos, edgelist= missing_edges, width = 0.5, edge_color = edge_colors)
    if ICM == True:
        plt.title('Simulated ICM with p= %.2f' % p)
    else:
        plt.title('Simulated TLM')

    plt.axis('off')

    visualize_set_actual(fname, pl_fname, convert_fname, nc_fname, all_nodes, pos)


def visualize_set(g, S, all_nodes):
    import networkx as nx
    node_color = []
    node_size = []
#    g = nx.subgraph(g, all_nodes)
    for v in g.nodes():
        if v in S:
            node_color.append('b')
            node_size.append(300)
        elif v in all_nodes:
            node_color.append('y')
            node_size.append(100)
        else:
            node_color.append('k')
            node_size.append(20)
#    node_size = [300 if v in S else 20 for v in g.nodes()]

    nx.draw(g, with_labels=True, node_color = node_color, node_size=node_size)


def visualize_set2(g, S, direct, convert):
    import networkx as nx

    node_color = []
    node_size = []
#    g = nx.subgraph(g, all_nodes)
    for v in g.nodes():
        if v in S:
            node_color.append('blue')
            node_size.append(300)  
        elif v in direct:
            if v in convert:
                node_color.append('green')
                node_size.append(200)
            else:
                node_color.append('yellowgreen')
                node_size.append(200)
        elif v in convert:
            node_color.append('red')
            node_size.append(200)
        else:
            node_color.append('k')
            node_size.append(20)
#    node_size = [300 if v in S else 20 for v in g.nodes()]
    nx.draw(g, with_labels=True, node_color = node_color, node_size=node_size)

def visualize_set_actual(fname, pl_fname, convert_fname, nc_fname, all_nodes, pos, min_color, max_color):
    import networkx as nx
    import matplotlib.pyplot as plt

    g = load_edgelist(fname)
    pl = strip(pl_fname)
    convert = strip(convert_fname)
    nc = strip(nc_fname)
    behavioral = convert + nc

    isolates = [x for x in all_nodes if x not in g.nodes()]
    print(isolates)
    g.add_nodes_from(isolates)
    # pos=nx.spring_layout(g)

    n = [x for x in all_nodes if x in nc]
    converts = [x for x in all_nodes if x in convert]
    missing = [x for x in all_nodes if x not in behavioral]
    missing = [x for x in missing if x not in pl]
  

    full = converts + n
    full_colors = [max_color]*len(converts)+[min_color+0.5]*len(n)
    n_colors = [min_color]*len(n)
    converts_colors = [max_color]*len(converts)
    missing_colors = ['#F0F2F4']*len(missing)
    print("v_min", min_color)
    print("v_max", max_color)

    blah = [max_color] * len(pl)
    # plt.subplot(311)
    plt.figure(1)
    nx.draw_networkx_labels(g,pos=pos,with_labels=True,font_size=8)

    nx.draw_networkx_nodes(g, pos= pos, nodelist= pl, node_color = 'r', node_size=300)
    nx.draw_networkx_nodes(g, pos=pos, nodelist=full, node_color = full_colors, cmap= plt.cm.PiYG,node_size=200)
    nx.draw_networkx_nodes(g, pos=pos, nodelist=missing, node_color = missing_colors, node_size = 200)

    # edges with missing nodes
    missing_edges = g.edges(missing)
    edge_colors = ['#E3E3E3'] * len(missing_edges)
    nx.draw_networkx_edges(g, pos=pos, width = 1.0)
    nx.draw_networkx_edges(g, pos=pos, edgelist= missing_edges, width = 0.5, edge_color = edge_colors)
    
    plt.title('Actual Network')
    plt.axis('off')
# sum up the neighborhood overlap for the entire PL set
# for each node, sum of overlap with every other PL nodee
def overlap(fname,pl_fname):
    import networkx as nx

    g = load_edgelist(fname)
    list_of_nodes = strip(pl_fname)
    list_of_nodes_2 = list_of_nodes[1:]
    overlap = []
    total = 0

    for node in list_of_nodes:
        for next in list_of_nodes_2:
            neighborhood = g.neighbors(next)
            neighborhood.append(next)
            overlap  = [x for x in g.neighbors(node) if x in neighborhood]
            print(overlap)
            node_neighbor = g.neighbors(node)
            node_neighbor.append(node)
            overlap = len(overlap)/float(len(node_neighbor+neighborhood))
            total = total + overlap
        if list_of_nodes_2: 
            list_of_nodes_2.pop(0)

    return total
            

def overlap2(fname,pl_list):
    import networkx as nx

    g = load_edgelist(fname)
    pl_list2 = pl_list[1:]
    overlap = []
    total = 0

    for node in pl_list:
        for next in pl_list2:
            neighborhood = g.neighbors(next)
            neighborhood.append(next)
            overlap  = [x for x in g.neighbors(node) if x in neighborhood]
            print(overlap)
            node_neighbor = g.neighbors(node)
            node_neighbor.append(node)
            overlap = len(overlap)/float(len(node_neighbor+neighborhood))
            total = total + overlap
        if pl_list2: 
            pl_list2.pop(0)

    return total

def overlap_onlyPL(fname,pl_list):
    import networkx as nx

    g = load_edgelist(fname)
    total = 0
    overlap = []

    for node in pl_list:
        overlap = [x for x in g.neighbors(node) if x in pl_list]
        print(overlap)
        overlap = len(overlap)/float(len(g.neighbors(node)))
        print(overlap)
        total = total + overlap
    return total

def redundancy(fname,pl_fname):
    import networkx as nx

    g= load_edgelist(fname)
    pl = strip(pl_fname)
    r = []
    penalty = []
    for each in pl:
        remove_this = [] 
        print("pl in question", each)
        ego = nx.ego_graph(g, each, radius = 1)
        n = len(ego.nodes())-1
        t = nx.edges(ego)
        print("edges", t)
        for node1, node2 in t:
            print("check", node1, node2)
            if node1 == each or node2 == each:
                remove_this.append((node1, node2))
                print("removed", node1, node2, each)
        print("removed this", remove_this)
        t = [x for x in t if x not in remove_this]     
        print("t", len(t))
        print("n", n)
        r.append(n-2*len(t)/float(n))
        penalty.append(2*len(t)/float(n))

    return penalty, sum(penalty), sum(r)


def percent_withPL(fname,pl_list):
    import networkx as nx

    g = load_edgelist(fname)
    total = 0
    overlap = []
    pl_list2  = pl_list[1:]
    total_degree = 0

    for node in pl_list:
        overlap = [x for x in g.neighbors(node) if x in pl_list2]
        overlap = len(overlap)
        # overlap = len(overlap)/float(len(g.neighbors(node)))
        print(overlap)
        skip = [x for x in pl_list if x not in pl_list2]
        print(skip)
        degree = len([x for x in g.neighbors(node) if x not in skip])
        print(degree)
        total_degree = total_degree + degree
        total = total + overlap
        if pl_list2: pl_list2.pop(0)

    
    return total/float(total_degree)

def spheres(fname, pl_fname, convert_fname, nc_fname):
    import networkx as nx

    g = load_edgelist(fname)
    pl_list = strip(pl_fname)
    convert = strip(convert_fname)
    nc = strip(nc_fname)
    total = nc+convert
    pl_list2 = pl_list[:]
    number_converted=0
    c={}
    n={}

    for node in pl_list:
        c[node] = [x for x in pl_list2 if x in g.neighbors(node)]
        n[node] = [x for x in pl_list2 if x not in g.neighbors(node)]

    s_c = {} 
    s_n = {}

    for key in c.keys():
        list_of_percent = []
        list_of_percent2 = []
        converted_original = [x for x in g.neighbors(key) if x in convert]
        only_original = [x for x in g.neighbors(key) if x not in pl_list]
        only_original = [x for x in only_original if x in total]

        by_itself_convert = len(converted_original)/float(len(only_original))
        list_of_percent.append(by_itself_convert)
        list_of_percent2.append(by_itself_convert)
        for each in c[key]:
            converted = [x for x in g.neighbors(each) if x in convert]
            converted = [x for x in converted if x not in pl_list]
            top = len(set(converted+converted_original))
            only = [x for x in g.neighbors(each) if x not in pl_list]
            only = [x for x in only if x in total]
            denom = len(set(only+only_original))
            if denom ==0:
                number_converted = 'denom 0'
            else:
                number_converted = top/float(denom)
            list_of_percent.append((each,number_converted, 'out of = %d' % denom))
        s_c[key] = list_of_percent
        for each in n[key]:
            if each == key:
                continue
            converted2 = [x for x in g.neighbors(each) if x in convert]
            converted2 = [x for x in converted2 if x not in pl_list]
            top = len(set(converted2 + converted_original))
            only = [x for x in g.neighbors(each) if x not in pl_list]
            only = [x for x in only if x in total]
            denom = len(set(only+only_original))
            if denom == 0:
                number_converted2 = 'denom 0'
            else:
                number_converted2 = top/float(denom)
            try:
                path = nx.shortest_path_length(g,source=key,target=each)
            except nx.exception.NetworkXNoPath:
                path = 'None'
            list_of_percent2.append((each, path,number_converted2, 'out of = %d' % denom))
        s_n[key] = list_of_percent2

    print(s_c)
    print(s_n)


def spheres_intersect(fname,pl_fname,convert_fname,nc_fname):
    import networkx as nx

    g = load_edgelist(fname)
    pl = strip(pl_fname)
    convert = strip(convert_fname)
    nc = strip(nc_fname)
    total = convert + nc
    only_neighbors_e = []
    only_neighbors_c = []
    counter_edge = 0
    counter_both = 0

    for each in total:
        try:
            pl_neighbors = [x for x in g.neighbors(each) if x in pl]
        except nx.exception.NetworkXError:
            continue
        number_pl_neighbors = len(pl_neighbors)
        if each in convert: 
            converted = 1
        else:
            converted = 0
        edge = 'no edge'
        if len(pl_neighbors):
            edge = len(pl_neighbors)
            counter_edge += 1
            if each in convert:
                counter_both += 1
            only_neighbors_e.append(edge)
            only_neighbors_c.append(converted)
    baseline_percent = len(convert)/float(len(total))
    edge_percent = counter_both/float(counter_edge)

    print(baseline_percent)
    print(edge_percent)
    print(only_neighbors_e)
    print(only_neighbors_c)

# returns list of all neighbors of a set of nodes 
def neighbors(g, S):
    import networkx as nx

    set_of_nbs = set()
    for node in S:
        for nb in g.neighbors(node):
            set_of_nbs.add(nb)
    return list(set_of_nbs)

def only_behavioral(direct, convert, nc):
    import networkx as nx

    behavioral = []
    for v in direct:
        if (v in convert) or (v in nc):
            behavioral.append(v)
    return behavioral

def visualize_communities(fname, convert_fname, nc_fname, pl_fname):
    import networkx as nx
    import community
    import numpy as np
    import random
    import matplotlib.pyplot as plt

    g = load_edgelist(fname)
    convert = strip(convert_fname)
    nc = strip(nc_fname)
    S = strip(pl_fname)

    total = convert + nc

    isolates = [x for x in total if x not in g.nodes()]
    g.add_nodes_from(isolates)

    part = community.best_partition(g)
    copy_part = part
    print(part)
    part = [part[x] for x in g.nodes()]
    print(part)
    com_names = np.unique(part)
    communities = []
    node_contig = range(len(g.nodes()))


    for i,c in enumerate(com_names):
        print(com_names)
        communities.append([])
        print(communities, i)
        print(communities[0], "communities 0")
        print(communities)
        test = [x for x in node_contig if part[x]==c]
        print(test)
        communities[i].extend([x for x in node_contig if part[x] == c])
    node_color = part
    pos = nx.layout.spring_layout(g, k=0.1)

    labels={}
    for node in S:
        labels[node] = node

    # pos = {}
    # centers = [[0, 0], [0, 1.25], [1.25, 0], [1.25, 1.25]]
    # for v in node_contig:
    #     print "part[v]", part[v]
    #     print centers[4][0]
    #     pos[v] = [centers[part[v]][0] + random.random() - 0.5, centers[part[v]][1] + random.random() - 0.5]

    # plt.figure(1)
    # plt.subplot(211)
    nx.draw(g, with_labels=False, node_color=node_color, pos=pos, node_size=50)
    nx.draw_networkx_labels(g,pos,labels,font_size=12,font_color='black')

    # if not S == None:
    #     node_colors = []
    #     node_sizes = []
    #     for v in g.nodes():
    #         if v in S:
    #             node_colors.append('red')
    #             node_sizes.append(100)
    #         else:
    #             node_colors.append('blue')
    #             node_sizes.append(50)
    # plt.subplot(212)
    # nx.draw(g, with_labels=False, node_color=node_colors, pos=pos, node_size = node_sizes)
    # nx.draw_networkx_labels(g,pos,labels,font_size=12,font_color='black')
    print(part)
    print(copy_part)

    plt.show()

def write_for_metis(g, fname):
    with open(fname, 'w') as f:
        f.write(str(g.number_of_nodes()) + ' ' + str(g.number_of_edges()) + '\n')
        for v in g.nodes():
            for u in g.neighbors(v):
                f.write(str(u+1) + ' ')
            f.write('\n')
  
def recursive_split(g):
    if len(g) == 1:
        return [g.nodes()[0]]
    return [recursive_split(g.subgraph(g.nodes()[:len(g)/2])), recursive_split(g.subgraph(g.nodes()[len(g)/2:]))]    
      
def recursive_partition(g, fname):
    print('CALL', len(g))
    import subprocess
    import numpy as np
    import networkx as nx
    if len(g) == 1:
        return [g.nodes()[0]]
    if g.number_of_edges() == 0:
        return recursive_split(g)
#    if len(g) == 2:
#        return [[g.nodes()[0]], [g.nodes()[1]]]
#    if len(g) == 3:
#        return [[[g.nodes()[0]], [g.nodes()[1]]], [g.nodes()[2]]]

    contig_g = nx.from_numpy_matrix(nx.to_numpy_matrix(g))
    write_for_metis(contig_g, fname)
    subprocess.call(['rm', fname + '.part.2'])
    subprocess.call(['./gpmetis', fname, '2'])
    labels = np.loadtxt(fname + '.part.2')
    nodes = np.array(g.nodes())
    print(len(nodes), len(labels))
    part1 = nodes[labels == 0]
    part2 = nodes[labels == 1]
    if len(part1) == 0:
        return recursive_split(g.subgraph(part2))      
    if len(part2) == 0:
        return recursive_split(g.subgraph(part1))
    return [recursive_partition(g.subgraph(part1), fname), recursive_partition(g.subgraph(part2), fname)]

def greedy_icm(g, budget, rr_sets = None, start_S = None):
    from rr_icm import make_rr_sets_cython, eval_node_rr
    import heapq
    num_nodes = len(g)
    allowed_nodes = range(num_nodes)
    if rr_sets == None:
        rr_sets = make_rr_sets_cython(g, 500, range(num_nodes))
    if start_S == None:
        S = set()
    else:
        S = start_S
    upper_bounds = [(-eval_node_rr(u, S, num_nodes, rr_sets), u) for u in allowed_nodes]    
    heapq.heapify(upper_bounds)
    starting_objective = 0
    #greedy selection of K nodes
    while len(S) < budget:
        val, u = heapq.heappop(upper_bounds)
        new_total = eval_node_rr(u, S, num_nodes, rr_sets)
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound
        if new_val >= -upper_bounds[0][0] - 0.1:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return S, starting_objective

def greedy(items, budget, f):
    '''
    Generic greedy algorithm to select budget number of items to maximize f.
    
    Employs lazy evaluation of marginal gains, which is only correct when f is submodular.
    '''
    import heapq
    upper_bounds = [(-f(set([u])), u) for u in items]    
    heapq.heapify(upper_bounds)
    starting_objective = f(set())
    S  = set()
    #greedy selection of K nodes
    while len(S) < budget:
        val, u = heapq.heappop(upper_bounds)
        new_total = f(S.union(set([u])))
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound
        if new_val >= -upper_bounds[0][0] - 0.1:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return S, starting_objective