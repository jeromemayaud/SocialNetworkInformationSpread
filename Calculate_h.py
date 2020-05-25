import networkx as nx
import numpy as np
import sys
from utils import *
from random import random
from random import sample
from random import choice
from random import randint
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import pandas as pd
from sklearn import metrics

############ FILE DIRECTORIES TO CHANGE ##############
fname = 'socialnetworkfile.txt'
pl_fname = 'peerleadersfile.txt'
#####################################################

g = load_edgelist(fname) #Load up edge list as a graph edge list
pl = strip(pl_fname) #List of peer leaders

h=[]
overlap = 0
for PeerLeader in pl:
    h.append(g.degree()[PeerLeader])
    for friend in g.neighbors(PeerLeader):
        if friend in pl: overlap = overlap + 1
overlap = overlap/2
h = (sum(h)-overlap)/float(sum(h))

print(h)