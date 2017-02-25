import sys
import numpy as np
from math import exp
from math import log
from copy import deepcopy
import time
#import matplotlib
#import matplotlib.pyplot as plt
import random
#from sklearn import metrics
from itertools import cycle
from align import get_matrix
from align import sequence_alignment
from align import res_similarity
from align import check_alignments


"""
Simulated annealing:
Let s = s0
1. Define temperature based on current iteration vs. maximum steps
2. Pick a new neighbor
If P(E(s), E(snew), T) > random(0,1) move to new state

"""
def read_alignments(f):
    with open(f,'r') as inf:
        alignments = eval(inf.read())
    return alignments

# Temperature function, set so that the temperature gradually decreases as we near the maximum iteration kmax.
def temp(k,kmax,score):
    u = (1 + 3*(4 - score)/4) # This function takes the difference in the current score from the optimal score and creates a parameter u,
    # which will amplify T if we take a large step in the wrong direction.
    T = u * 200 * 0.83**(80*k/kmax)
    return T

# Probability function. P is set to 1 if the new score is better than the previous, otherwise it is
# defined as the exponential of the negative difference between the old score and new score over T.
def P(score,newscore,T):
    if score < newscore:
        P = 1
    else:
        #P = 0
        P = exp(-(score - newscore)/T)
    return P
"""
def roc_calc(binary,scores):
    fpr = []
    tpr = []
    min_score = min(scores)
    max_score = max(scores)
    thr = np.linspace(min_score, max_score, 30)
    FP=0
    TP=0
    N = sum(y)
    P = len(y) - N

    for (i, T) in enumerate(thr):
        for i in range(0, len(score)):
            if (score[i] > T):
                if (y[i]==1):
                    TP = TP + 1
                if (y[i]==0):
                    FP = FP + 1
        fpr.append(FP/float(N))
        tpr.append(TP/float(P))
        FP=0
        TP=0
    return fpr, tpr, thr
"""

# Function to find the next neighbor. With 288 positions capable of being altered, changing one at a time
# is probably too slow, but changing too many will make it difficult for the algorithm to converge on
# the optimal solution.
def neighbor(matrix):
    altmatrix = deepcopy(matrix)
    for k in range(0,3): #randomize 3 places on the matrix
        # pick a random spot in the matrix, but ignore the last row and column since gap penalty isn't being modified.
        i = random.randint(0,22)
        j = random.randint(0,22)
        if float(matrix[i+1][j]) <= abs(12):
            x = float(matrix[i+1][j]) + 3
            y = float(matrix[i+1][j]) - 3
        elif float(matrix[i+1][j]) < 0:
            x = -15
            y = -9
        elif float(matrix[i+1][j]) > 0:
            x = 15
            y = 9
        altmatrix[i+1][j] = random.uniform(x,y) # score not restricted to integer
        altmatrix[j+1][i] = altmatrix[i+1][j] # maintain symmetry
    return altmatrix

# score function, where the score is the sum of the true positive rates for false positive rates of
# 0, 0.1, 0.2, and 0.3.
def obj_score(poslist,neglist,matrix):
    #pos = get_scorelist(poslist,matrix)
    #neg = get_scorelist(neglist,matrix)
    pos,neg = get_scorelists(matrix)
    p00 = np.percentile(neg,100)
    p01 = np.percentile(neg,90)
    p02 = np.percentile(neg,80)
    p03 = np.percentile(neg,70)
    plist = [p00,p01,p02,p03]
    score = 0
    for i in plist:
        poscount = 0
        for j in pos:
            if j > i:
                poscount += 1
        tpr = poscount / len(pos)
        score += tpr
    return score

def get_scorelists(matrix):
    scorelistpos = []
    scorelistneg = []
    with open("Pospairs.txt","r") as pairlist:
        for line in pairlist:
            pair = []
            if line [0] != "#":
                pair.append(line.split())
                print(pair)
                align = sequence_alignment(pair[0][0],pair[0][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                scorelistpos.append(align[2])
    with open("Negpairs.txt","r") as pairlist:
        for line in pairlist:
            pair = []
            if line[0] != "#":
                pair.append(line.split())
                print(pair)
                align = sequence_alignment(pair[0][0],pair[0][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                scorelistneg.append(align[2])
    return scorelistpos,scorelistneg

def get_scorelist(alignlist,matrix):
    scorelist = []
    for i in range(0,len(alignlist)):
        check = check_alignments(alignlist[i][0],alignlist[i][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
        scorelist.append(check)
    return scorelist

def simulated_annealing(alignlistpos,alignlistneg,input_matrix,kmax): # runs the actual algorithm
    s0 = input_matrix
    maxscore = 0
    newscore = 0
    for k in range(0,kmax):
        oldscore = obj_score(alignlistpos,alignlistneg,s0)
        T = temp(k,kmax,oldscore) # Get temperature for iteration "k"
        #print("temp",T)
        stemp = s0
        #print(stemp)
        snew = neighbor(s0)
        newscore = obj_score(alignlistpos,alignlistneg,snew)
        if newscore > maxscore:
            maxscore = newscore
            smax = deepcopy(snew)
            iteration = k
        print(oldscore,newscore)
        #print("prob:",P(oldscore,newscore,T))
        if P(oldscore,newscore,T) >= random.uniform(0,1):
            s0 = snew
            print("-------Accepted!-------")
        else:
            print('-------Rejected-------')
        if newscore == 4: # Just in case :)
            break
    return smax,s0,newscore,maxscore,iteration,kmax

def write_file(matrix,score,kmax):
    current_time = time.strftime("%m.%d_t_%H:%M",time.localtime())
    output_name = 'results/m_kmax_{}_score_{:03.2f}_d_{}'.format(kmax,score,current_time)
    output_file = open('{}.txt'.format(output_name),'w')
    for index in range(0,len(matrix)):
        for item in matrix[index]:
            output_file.write('%s  '%item)
        output_file.write('\n')
    return output_name

def roc_curve(scorelistpos,scorelistneg):
    pos = [1] * len(scorelistpos)
    neg = [0] * len(scorelistneg)
    y_true = pos + neg
    scores = scorelistpos + scorelistneg
    fpr, tpr, thresholds = roc_calc(y_true,scores)
    roc_auc = metrics.auc(fpr,tpr)
    return (fpr,tpr,thresholds,roc_auc)



matrix = get_matrix('PAM100')
all_pos = read_alignments('posaligns.txt')
all_neg = read_alignments('negaligns.txt')

run_annealing = simulated_annealing(all_pos,all_neg,matrix,1000)
#print(run_annealing)
wr = write_file(run_annealing[0],run_annealing[3],run_annealing[5])
newmat = get_matrix('{}.txt'.format(wr))
scorecheck = obj_score(all_pos,all_neg,newmat)
"""
#print(scorecheck)
m = ['{}.txt'.format(wr),'PAM100']
fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
for i in m:
    print(i)
    scorelistpos = get_scorelist(all_pos,get_matrix(i))
    scorelistneg = get_scorelist(all_neg,get_matrix(i))
    fpr[i],tpr[i],thresholds[i],roc_auc[i] = roc_curve(scorelistpos,scorelistneg)
colors = cycle(['aqua','darkorange','cornflowerblue','darkred','black'])
lw = 2
for i, color in zip(m,colors):
    plt.plot(fpr[i],tpr[i],color = color,lw = lw,label = '{0} (area = {1:0.2f})'.format(i,roc_auc[i]))
plt.plot([0,1],[0,1],'k--',lw=lw)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver operating characteristics of normalized substitution matrices \n gap start penalty = 11, gap extension penalty = 1')
plt.legend(loc = "lower right")
plt.savefig('{}.png'.format(wr))
plt.close()
#plt.show()
"""
"""
Test the annealing vs. original matrix by using it to actually realign sequences
"""
"""
l = ['results/m_kmax_10000_score_3.26_d_02.24_t_15:19.txt','PAM100']
fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
fp70 = dict()
for m in l:
    print("Generating ROC data for",m,"matrix")
    matrix = get_matrix(m)
    scorelistpos = []
    scorelistneg = []
    with open("Pospairs.txt","r") as pairlist:
        for line in pairlist:
            pair = []
            if line [0] != "#":
                pair.append(line.split())
                print(pair)
                align = sequence_alignment(pair[0][0],pair[0][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                scorelistpos.append(align[2])
    with open("Negpairs.txt","r") as pairlist:
        for line in pairlist:
            pair = []
            if line[0] != "#":
                pair.append(line.split())
                print(pair)
                align = sequence_alignment(pair[0][0],pair[0][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                scorelistneg.append(align[2])
    # For each set of scores, generate ROC data and add it to a dictionary:
    fpr[m],tpr[m],thresholds[m],roc_auc[m] = roc_curve(scorelistpos,scorelistneg)
# Now to plot the ROC curves:
colors = cycle(['aqua','darkorange','cornflowerblue','darkred','black'])
current_time = time.strftime("%m.%d_%H:%M",time.localtime())
lw = 2
for i, color in zip(l,colors):
    plt.plot(fpr[i],tpr[i],color = color,lw = lw,label = '{0} (area = {1:0.2f})'.format(i,roc_auc[i]))
plt.plot([0,1],[0,1],'k--',lw=lw)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver operating characteristics of normalized substitution matrices \n gap start penalty = 11, gap extension penalty = 1')
plt.legend(loc = "lower right")
plt.savefig('{}_q2compare.png'.format(wr))
plt.show()
"""
