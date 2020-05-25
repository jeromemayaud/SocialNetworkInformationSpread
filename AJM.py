import networkx as nx
import numpy as np
from utils import *
import sys
from random import random
from random import sample
from random import choice
from random import randint
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import sklearn.metrics
from scipy import interp

############ FILE DIRECTORIES TO CHANGE ##############
fname = 'socialnetworkfile.txt'
pl_fname = 'peerleadersfile.txt'
convert_fname = 'confirmedconversationsfile.txt'
nc_fname = 'noconversationsfile.txt'
all_nodes_fname = 'allnodesfile.txt'
#####################################################

g = load_edgelist(fname) #Load up edge list as a graph edge list
pl = strip(pl_fname) #List of peer leaders
convert = strip(convert_fname)
nc = strip(nc_fname)
all_nodes = strip_demo(all_nodes_fname)
	
def social_distance(g, pl):
    d={}

    for node in pl:
        d[node] = nx.shortest_path_length(g,source=node)
    return d

def simulate_jump_random(g, pl, p):
    ###### PREVIOUS DATASETS ######
   
    h = 0.87 #Acquired from running 'Calculate_h' file
    
    converted = pl[:]
    dead = []
    not_converted = [x for x in g.nodes() if x not in pl]

    left = []
    not_converted = [x for x in not_converted if x not in left]

    d = social_distance(g, pl)

    for influencer in pl:
        if 1/float(g.degree()[influencer]*h) > 1:
            activation = 1
        else:
            activation = int(np.random.geometric(p=1/float((g.degree()[influencer])*h), size = 1))
        if set(dead) == set(converted):
            break
        if influencer not in dead:
            for one in range(activation):
                particular_eth = 0
                particular_s = 0
                if len(not_converted):
                    prob = []
                    for node in not_converted:
                        particular_deg = 1/float((1+g.degree()[node]))
                        try:
                            particular_dis = 1/d[influencer][node]
                        except KeyError:
                            particular_dis = 0.1
                        prob.append(particular_dis+particular_deg+particular_eth+particular_s)
                    total = sum(prob)
                    picked = random()
                    for node, probability in zip(not_converted, prob):
                        picked = picked - probability/float(total)
                        if picked <= 0:
                            jumped_node = node
                            break
                else: break
                if random() < (p+p/float((1+g.degree()[jumped_node]))): #probability that converted
                    not_converted.remove(jumped_node)
                    converted.append(jumped_node)
                    
            dead.append(influencer)

    return(converted+pl)
    print("converted+pl", converted+pl)

def main(g, all_nodes, pl, convert, nc, p, number):
    r = {}
    new_range = np.linspace(0,1,100).tolist()
    
    all_nodes1 = []
    for sublist in all_nodes:
        all_nodes1.append(sublist[0])
        
    all_nodes_except_PL = [x for x in all_nodes1 if x not in pl]
	
    for node in all_nodes_except_PL:
        r[node] = 0

    isolates = [x for x in all_nodes1 if x not in g.nodes()]
    g.add_nodes_from(isolates)
    
    binary_index = []
    for converted in all_nodes_except_PL:
        if converted in convert:
            binary_index.append(1)
        elif converted in nc:
            binary_index.append(0)
        else:
            binary_index.append(-1)
    
    FPR_total = []
    TPR_total = []

    sensitivity, specificity = 0, 0
    running_s = []
    running_p = []
    
    running_binary_sim = [0]*len(all_nodes_except_PL)
    activations = np.zeros((number, len(all_nodes_except_PL)))
    number_of_predicted_positives = np.zeros(number)
    
    for i in range(number):
        converted_list = simulate_jump_random(g, pl, p)
        converted_list = [x for x in converted_list if x not in pl]
        binary_sim = []
        for idx, each in enumerate(all_nodes_except_PL):
            if each in converted_list:
                binary_sim.append(1)
                activations[i, idx] = 1
            else:
                binary_sim.append(0)
        running_binary_sim = [sum(x) for x in zip(binary_sim, running_binary_sim)]
        number_of_predicted_positives[i] = sum(binary_sim)
        
    running_binary_sim = [x/float(number) for x in running_binary_sim]
    
    running_binary_sim = np.array(running_binary_sim)
    binary_index = np.array(binary_index)
    have_data = np.where(binary_index != -1)[0]
    
    max_auc_sklearn = sklearn.metrics.roc_auc_score(binary_index[have_data], running_binary_sim[have_data])
    roc_curve = sklearn.metrics.roc_curve(binary_index[have_data], running_binary_sim[have_data])
    
    avg_predicted_positives = np.round(np.mean(number_of_predicted_positives))
    observed_positives = np.size(np.where(binary_index == 1))
        
    print('AUC:', max_auc_sklearn)
    
    return max_auc_sklearn, avg_predicted_positives, observed_positives, roc_curve

def multiple_runs(g, all_nodes, pl, convert, nc, p, number, number_of_simulations):
    
    running_auc_results = []
    avg_predicted_positives_list = []
    observed_positives_list = []
    ratio_predicted_to_observed_positives_list = []
    running_fpr_results = []
    running_tpr_results = []
    
    for i in range (number_of_simulations):
        max_auc_sklearn, avg_predicted_positives, observed_positives, roc_curve = main(g, all_nodes, pl, convert, nc, p, number)
        
        ratio_predicted_to_observed_positives = avg_predicted_positives/observed_positives
        
        print('Ratio of predicted to observed positives:', ratio_predicted_to_observed_positives)
        
        running_auc_results.append(max_auc_sklearn)
        avg_predicted_positives_list.append(avg_predicted_positives)
        observed_positives_list.append(observed_positives)
        ratio_predicted_to_observed_positives_list.append(ratio_predicted_to_observed_positives)
        running_fpr_results.append(roc_curve[0])
        running_tpr_results.append(roc_curve[1])
        
        print("*****", i+1, " simulations run *****")

    auc_mean = np.mean(running_auc_results)
    auc_stdev = np.std(running_auc_results)
    avg_predicted_positives_mean = np.mean(avg_predicted_positives_list)
    observed_positives_mean = np.mean(observed_positives_list)
    ratio_predicted_to_observed_positives_mean = np.mean(ratio_predicted_to_observed_positives_list)
    ratio_predicted_to_observed_positives_std = np.std(ratio_predicted_to_observed_positives_list)

#Find macro-average of ROC curves
    mean_fpr = np.unique(np.concatenate([running_fpr_results[i] for i in range(number_of_simulations)])) #Aggregate all false positive rates
    mean_tpr = np.zeros_like(mean_fpr) #Interpolate all ROC curves at these points
    temporary_tpr = np.zeros_like(mean_fpr) 
    temporary_count = np.zeros_like(mean_fpr)
    
    for k in range(np.size(mean_fpr)):
        fpr_value = mean_fpr[k]
        for i in range(number_of_simulations):
            temporary_tpr[k] += np.sum(running_tpr_results[i][np.where(running_fpr_results[i] == fpr_value)])
            temporary_count[k] += np.size(np.where(running_fpr_results[i] == fpr_value))
    mean_tpr = temporary_tpr/temporary_count
 
    return auc_mean, auc_stdev, ratio_predicted_to_observed_positives_mean, ratio_predicted_to_observed_positives_std, avg_predicted_positives_mean, observed_positives_mean, running_fpr_results, running_tpr_results, mean_fpr, mean_tpr

########## THIS IS WHAT RUNS 'MAIN' ###############
auc_mean, auc_stdev, ratio_predicted_to_observed_positives_mean, ratio_predicted_to_observed_positives_std, avg_predicted_positives_mean, observed_positives_mean, running_fpr_results, running_tpr_results, mean_fpr, mean_tpr = multiple_runs(g, all_nodes, pl, convert, nc, 0.1, 100, 10)
##################################################	
