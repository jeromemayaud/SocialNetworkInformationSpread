import numpy as np
from sklearn import metrics
import pandas as pd

fname = 'socialnetworkfile.txt'
df = pd.read_csv(fname,delimiter="\t", header=None)

Y_test = np.array(df)[:,0]
Y_test_prob_predict = np.array(df)[:, 3] #This is the column you change to change which model is tested


#### CALCULATE FPR and TPR FROM DIFFERENT THRESHOLDS, FOR THE AUROC ##########
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_prob_predict, pos_label=1)
auc_mdl = metrics.auc(fpr, tpr)
##############################################################################


def Find_Optimal_Cutoff(target, predicted):
    
    #Find the optimal probability cutoff point for a classification model related to event rate
    #*Parameters:
    #target : Matrix with dependent or target data, where rows are observations
    #predicted : Matrix with predicted data, where rows are observations
    #*Returns: list type, with optimal cutoff value

    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr + (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf - 0).abs().argsort()[::-1]]
    return list(roc_t['threshold'])

# FIND OPTIMAL PROBABILITY CUTOFF POINT 
bestCutOff_mdl = Find_Optimal_Cutoff(Y_test, Y_test_prob_predict)[0]

Y_test_predict = pd.DataFrame(Y_test_prob_predict).loc[:, 1].map(lambda x: 1 if x > bestCutOff_mdl else 0)

precision_mdl = metrics.precision_score(Y_test, Y_test_predict, pos_label=1)

recall_mdl = metrics.recall_score(Y_test, Y_test_predict, pos_label=1)
