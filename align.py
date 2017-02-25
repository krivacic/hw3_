import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle

""" Usage: python align.py <substitution matrix/all> <gap start penalty (negative number)> <gap extension penalty (negative number)> """


def get_matrix(matrix):
    with open(matrix,"r") as f:
        matrix = []
        for line in f:
            if line[0] != "#":
                matrix.append(line.split())
    return matrix

def res_similarity(aa1,aa2,matrix):
    """
    Input: two amino acids.
    Output: similarity of the amino acid pair from the similarity matrix.
    """

    i = matrix[0].index(aa1)
    j = matrix[0].index(aa2)

    s = float(matrix[i+1][j])

    return s

def sequence_alignment(residues_a,residues_b,gapstart,gap,matrix):

    """ This function aligns two amino acid sequences.

    Input: two lists of residues, a gap penalty, and a gap start penalty.
    Output: two lists of aligned sequences and a score.

    """
    reslist_a = ["*"]
    reslist_b = ["*"]
    fa = open(residues_a)
    fb = open(residues_b)
    # populate residue lists
    for line in fa:
        if line[0] != ">":
            for c in line:
                if c != "\n":
                    reslist_a.append(c)
    for line in fb:
        if line[0] != ">":
            for c in line:
                if c != "\n":
                    reslist_b.append(c)
    len1 = len(reslist_a)
    len2 = len(reslist_b)
    # For this algorithm, we need three scoring matrices to find the best possible alignment.
    # F is the scoring matrix, GH keeps track of horizontal gaps, and GV keeps track of vertical gaps.
    F = np.zeros(shape = (len1,len2))
    GH = np.zeros(shape = (len1,len2))
    GV = np.zeros(shape = (len1,len2))
    # Tracking matrices to reconstruct sequence:
    traceF = np.zeros(shape = (len1,len2))
    traceGH = np.zeros(shape = (len1,len2))
    traceGV = np.zeros(shape = (len1,len2))
    # define gap penalty "d" and gap open penalty "gap_open":
    gap_open = gapstart
    d = gap
    # Now we fill the matrix in with score values. For each box, we test the score for
    # a match, as well as a deletion and insertion, and pick whichever score is the highest.
    for i in range (1,len1):
        for j in range(1,len2):
            # GH always moves horizontally - either from F[i,j-1] or G[i,j-1].
            GH[i,j] = max(F[i,j-1] + gap_open, GH[i,j-1] + d) #gap matrix horizontal
            #traceGH[i,j] = np.argmax([F[i,j-1]+ gap_open, GH[i,j-1] + d])
            # GV always moves vertically - either from F[i-1,j] or G[i-1,j].
            GV[i,j] = max(F[i-1,j] + gap_open, GV[i-1,j] + d) #gap matrix vertical
            #traceGV[i,j] = np.argmax([F[i-1,j] + gap_open, GV[i-1,j] + d])
            match = res_similarity(reslist_a[i],reslist_b[j],matrix)
            # Find best score possible for scoring matrix F at point [i,j]
            # F always moves diagonally, but it can originate from a vertical or horizontal gap (GV or GH), or from a previous match.
            F[i,j] = max(F[i-1,j-1] + match, GH[i-1,j-1] + match, GV[i-1,j-1] + match, 0)
            #traceF[i,j] = np.argmax([F[i-1,j-1] + match, GH[i-1,j-1] + match, GV[i-1,j-1] + match,0])
    score = np.amax(F)
    start = np.unravel_index(np.argmax(F),F.shape)
    alignment_a = []
    alignment_b = []
    i = start[0]
    j = start[1]
    """
    # "Walking back:"

    # The "current" variable will tell us whether we're in an F matrix, a GH matrix, or a GV matrix, allowing us to determine
    # whether we should append a gap for either of the sequences, and which direction to travel for the next iteration.
    # Start by assuming we're not ending on a gap (should always be true since gap means negative points)
    # That is, we must start by looking at the score matrix F. Let 0 denote matrix F, 1 denote matrix GH, and 2 denote matrix GV.
    current = 0
    stop = False
    while i > 0 or j > 0:
        if stop == True: # Stops if I find a matrix that is equal to 0 (in any of the 3 matrices)
            break
        if current == 3: # Stops if I hit an index of 3, which would also mean I've reached a 0 matrix; I could probably take this out but I'm not gonna risk it.
            break
        # If we're in the F matrix, insert residues for both sequences, check the F trace matrix to determine how to handle the next iteration,
        # then walk backwards diagonally.
        if current == 0:
            if F[i,j] <= 0: # checks for 0 square so we know to stop.
                stop = True
                break
            alignment_a.insert(0,reslist_a[i])
            alignment_b.insert(0,reslist_b[j])
            current = traceF[i,j]
            i = i - 1
            j = j - 1
        # If we're in the GH matrix, we insert one gap and one residue, check the GH trace matrix to determine how to handle the next iteration,
        # then walk back horizontally.
        elif current == 1:
            if GH[i,j] <= 0: # checks for 0 square so we know to stop.
                stop = True
                break
            alignment_a.insert(0,"*")
            alignment_b.insert(0,reslist_b[j])
            current = traceGH[i,j]
            j = j - 1
        # If we're in the GV matrix, we insert one residue and one gap, check the GV trace matrix to determine how to handle the next iteration,
        # then walk back horizontally.
        elif current == 2:
            if GV[i,j] <= 0: # checks for 0 square so we know to stop.
                stop == True
                break
            alignment_a.insert(0,reslist_a[i])
            alignment_b.insert(0,"*")
            current = traceGV[i,j]
            # I used "2" to represent the GV matrix because that's how you get to it from traceF[i,j],
            # but since traceGV only uses values 0 and 1, I have to convert 1 to 2 in order to continue to move
            # vertically.
            if current == 1:
                current = 2
            i = i - 1
    """
    return (alignment_a,alignment_b,score,len1-1,len2-1)


def write_alignments(list,filename):
    out = open(filename, 'w')
    out.write(str(list))
    out.close()


def check_alignments_fromfile(sequence_a,sequence_b,gapstart,gap,matrix):
    fa = open(sequence_a)
    fb = open(sequence_b)
    reslist_a = []
    reslist_b = []
    # populate residue lists
    for line in fa:
        if line[0] != ">":
            for c in line:
                if c != "\n":
                    if c == "-":
                        reslist_a.append("*")
                    else:
                        reslist_a.append(c)
    for line in fb:
        if line[0] != ">":
            for c in line:
                if c != "\n":
                    if c== "-":
                        reslist_b.append("*")
                    else:
                        reslist_b.append(c)
    scorelist = []
    print(reslist_a)
    print(reslist_b)
    assert len(reslist_a) == len(reslist_b)
    gaplen = 0
    for i in range(0,len(reslist_a)):
        if reslist_a[i] != '*' and reslist_b[i] != '*':
            score = res_similarity(reslist_a[i],reslist_b[i],matrix)
            scorelist.append(score)
            gaplen = 0
        elif reslist_a[i] == '*' or reslist_b[i] == '*':
            gaplen += 1
            print(gaplen)
            if gaplen == 1:
                score = gapstart
                scorelist.append(score)
            else:
                score = gap
                scorelist.append(score)
        print(reslist_a[i],reslist_b[i],score)
    final = sum(scorelist)
    return final

def check_alignments(sequence_a,sequence_b,gapstart,gap,matrix):
    finalscore = 0
    assert len(sequence_a) == len(sequence_b)
    gaplen = 0
    for i in range(0,len(sequence_a)):
        if sequence_a[i] != '*' and sequence_b[i] != '*':
            score = res_similarity(sequence_a[i],sequence_b[i],matrix)
            finalscore += score
            gaplen = 0
        elif sequence_a[i] == '*' or sequence_b[i] == '*':
            gaplen += 1
            #print(gaplen)
            if gaplen == 1:
                finalscore += gapstart
            else:
                finalscore += gap
        #print(sequence_a[i],sequence_b[i],score)
    #print(len(scorelist))
    return finalscore

#matrix = get_matrix(sys.argv[1])

#align = sequence_alignment("sequences/prot-0018.fa","sequences/prot-0198.fa",-10,-1,matrix)
#align = sequence_alignment("sequences/test1","sequences/test2",-10,-1,matrix)
#print(align)
#check = check_alignments(align[0],align[1],-10,-1,matrix)
#check = check_alignments_fromfile('testing','testing2',-10,-1,matrix)
#print("Alignment score:",align[2])
#print("Score check:",check)
"""
The following three blocks of code generate ROC curves for:
    1) all provided substitution matrices
    2) all provided substitution matrices with scores normalized to the smallest sequences
    3) the best provided substitution matrix with raw scores vs. normalized scores
depending on the system input.
"""
"""
if sys.argv[1] == 'best_matrix':
    l = ['BLOSUM50','BLOSUM62','MATIO','PAM100','PAM250']
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
        pos = [1] * len(scorelistpos)
        neg = [0] * len(scorelistneg)
        y_true = pos + neg
        scores = scorelistpos + scorelistneg
        fpr[m],tpr[m],thresholds[m] = metrics.roc_curve(y_true,scores)
        roc_auc[m] = metrics.auc(fpr[m],tpr[m])
        cutoff = np.percentile(scorelistpos,30)
        negcount = 0
        for score in scorelistneg:
            if score > cutoff:
                negcount += 1
        fp70[m] = negcount / len(scorelistneg)
    # Now to plot the ROC curves:
    lw = 2
    colors = cycle(['aqua','darkorange','cornflowerblue','darkred','black'])
    for i, color in zip(l,colors):
        plt.plot(fpr[i],tpr[i],color = color,lw = lw,label = '{0} (area = {1:0.2f}, \nfalse pos rate at TPR of 0.7 = {2})'.format(i,roc_auc[i],fp70[i]))
    plt.plot([0,1],[0,1],'k--',lw=lw)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver operating characteristics of substitution matrices \n gap start penalty = 11, gap extension penalty = 1')
    plt.legend(loc = "lower right")
    plt.show()

elif sys.argv[1] == 'best_matrix_normalized':
    l = ['BLOSUM50','BLOSUM62','MATIO','PAM100','PAM250']
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    fp70 = dict()
    for m in l:
        print("Generating ROC data for",m,"matrix")
        matrix = get_matrix(m)
        scorelistpos_norm = []
        scorelistneg_norm = []
        with open("Pospairs.txt","r") as pairlist:
            for line in pairlist:
                pair = []
                if line [0] != "#":
                    pair.append(line.split())
                    print(pair)
                    align = sequence_alignment(pair[0][0],pair[0][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                    scorelistpos_norm.append(align[2]/min(align[3],align[4]))
        with open("Negpairs.txt","r") as pairlist:
            for line in pairlist:
                pair = []
                if line[0] != "#":
                    pair.append(line.split())
                    print(pair)
                    align = sequence_alignment(pair[0][0],pair[0][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                    scorelistneg_norm.append(align[2]/min(align[3],align[4]))
        # For each set of scores, generate ROC data and add it to a dictionary:
        pos = [1] * len(scorelistpos) # creates binary data set to tell whether values are positives or negatives
        neg = [0] * len(scorelistneg)
        y_true = pos + neg
        scores = scorelistpos + scorelistneg
        print(scores)
        fpr[m],tpr[m],thresholds[m] = metrics.roc_curve(y_true,scores)
        roc_auc[m] = metrics.auc(fpr[m],tpr[m])
        cutoff = np.percentile(scorelistpos,30)
        negcount = 0
        for score in scorelistneg:
            if score > cutoff:
                negcount += 1
        fp70[m] = negcount / len(scorelistneg)
    # Now to plot the ROC curves:
    lw = 2
    colors = cycle(['aqua','darkorange','cornflowerblue','darkred','black'])
    for i, color in zip(l,colors):
        plt.plot(fpr[i],tpr[i],color = color,lw = lw,label = '{0} (area = {1:0.2f}, \nfalse pos rate at TPR of 0.7 = {2})'.format(i,roc_auc[i],fp70[i]))
    plt.plot([0,1],[0,1],'k--',lw=lw)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver operating characteristics of normalized substitution matrices \n gap start penalty = 11, gap extension penalty = 1')
    plt.legend(loc = "lower right")
    plt.show()


elif sys.argv[1] == 'compare_normalized':
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    fp70 = dict()
    matrix = get_matrix('PAM100')
    scorelistpos = []
    scorelistpos_norm = []
    scorelistneg = []
    scorelistneg_norm = []
    with open("Pospairs.txt","r") as pairlist:
        for line in pairlist:
            pair = []
            if line [0] != "#":
                pair.append(line.split())
                print(pair)
                align = sequence_alignment(pair[0][0],pair[0][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                scorelistpos.append(align[2])
                scorelistpos_norm.append(align[2]/min(align[3],align[4]))
    with open("Negpairs.txt","r") as pairlist:
        for line in pairlist:
            pair = []
            if line[0] != "#":
                pair.append(line.split())
                print(pair)
                align = sequence_alignment(pair[0][0],pair[0][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                scorelistneg.append(align[2])
                scorelistneg_norm.append(align[2]/min(align[3],align[4]))
    # For each set of scores, generate ROC data and add it to a dictionary:
    pos = [1] * len(scorelistpos)
    neg = [0] * len(scorelistneg)
    y_true = pos + neg
    scores = scorelistpos + scorelistneg
    scores_norm = scorelistpos_norm + scorelistneg_norm
    l = ['Raw scores','Normalized scores']
    for m in l:
        if m == 'Raw scores':
            fpr[m],tpr[m],thresholds[m] = metrics.roc_curve(y_true,scores)
            roc_auc[m] = metrics.auc(fpr[m],tpr[m])
            cutoff = np.percentile(scorelistpos,30)
            negcount = 0
            for score in scorelistneg:
                if score > cutoff:
                    negcount += 1
            fp70[m] = negcount / len(scorelistneg)
        if m == 'Normalized scores':
            fpr[m],tpr[m],thresholds[m] = metrics.roc_curve(y_true,scores_norm)
            roc_auc[m] = metrics.auc(fpr[m],tpr[m])
            cutoff = np.percentile(scorelistpos_norm,30)
            negcount = 0
            for score in scorelistneg_norm:
                if score > cutoff:
                    negcount += 1
            fp70[m] = negcount / len(scorelistneg_norm)
    # Now to plot the ROC curves:
    lw = 2
    colors = cycle(['aqua','darkorange','cornflowerblue','darkred','black'])
    for i, color in zip(l,colors):
        plt.plot(fpr[i],tpr[i],color = color,lw = lw,label = '{0} (area = {1:0.2f}, \nfalse pos rate at TPR of 0.7 = {2})'.format(i,roc_auc[i],fp70[i]))
    plt.plot([0,1],[0,1],'k--',lw=lw)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver operating characteristics of PAM100 matrix with raw and normalized scores \n gap start penalty = 11, gap extension penalty = 1')
    plt.legend(loc = "lower right")
    plt.show()
"""
"""
# This block of code writes the alignments to a file so that we can quickly optimize the scoring matrix
elif sys.argv[1] == 'write_alignments':
    matrix = get_matrix('PAM100')
    scorelistpos = []
    posdic = []
    negdic = []
    with open("Pospairs.txt","r") as pairlist:
        for line in pairlist:
            pair = []
            if line[0] != "#":
                pair.append(line.split())
                print(pair[0])
                align = sequence_alignment(pair[0][0],pair[0][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                check = check_alignments(align[0],align[1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                print(align[2],check)
                assert align[2] == check # make sure my alignments are correct
                posdic.append([align[0],align[1]])
    scorelistneg = []
    with open("Negpairs.txt","r") as pairlist:
        for line in pairlist:
            pair = []
            if line[0] != "#":
                pair.append(line.split())
                print(pair[0])
                align = sequence_alignment(pair[0][0],pair[0][1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                check = check_alignments(align[0],align[1],float(sys.argv[2]),float(sys.argv[3]),matrix)
                print(align[2],check)
                assert align[2] == check
                negdic.append([align[0],align[1]])
    write_alignments(posdic,'posaligns.txt')
    write_alignments(negdic,'negaligns.txt')

"""
"""
The following block of code is for optimizing gap start and gap continue penalties.
It will be commented out unless needed.
"""

"""
# this list will be populated with three lists: gap penalty, gap start penalty, and false positive fraction

gapdata = [[],[],[]]
gap_pen_list = [-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,-0,3,-0.1]
matrix = get_matrix(sys.argv[1])
for j in range(-20,-1,1):
    for i in gap_pen_list:
        gapdata[0].append(i)
        gapdata[1].append(j)
        scorelistpos = []
        with open("Pospairs.txt","r") as pairlist:
            for line in pairlist:
                pair = []
                if line[0] != "#":
                    pair.append(line.split())
                    print(pair[0])
                    align = sequence_alignment(pair[0][0],pair[0][1],j,i,matrix)
                    scorelistpos.append(align[2])
            print(scorelistpos)
        scorelistneg = []
        with open("Negpairs.txt","r") as pairlist:
            for line in pairlist:
                pair = []
                if line[0] != "#":
                    pair.append(line.split())
                    print(pair[0])
                    align = sequence_alignment(pair[0][0],pair[0][1],i,j,matrix)
                    scorelistneg.append(align[2])
            print(scorelistneg)

        print("mean score positive:",sum(scorelistpos)/len(scorelistpos))
        print("mean score negative:",sum(scorelistneg)/len(scorelistneg))
        print("percentile",np.percentile(scorelistpos,30))
        negcount = 0
        cutoff = np.percentile(scorelistpos,30)
        for score in scorelistneg:
            if score > cutoff:
                negcount += 1
        fpf = negcount / len(scorelistneg)
        gapdata[2].append(fpf)
        print(gapdata)
print(gapdata)

# The following list is the data from above code block:
#gapdata = [[-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1], [0.4, 0.36, 0.36, 0.38, 0.38, 0.4, 0.28, 0.28, 0.32, 0.24, 0.28, 0.26, 0.32, 0.36, 0.38, 0.4, 0.36, 0.36, 0.38, 0.38, 0.4, 0.28, 0.28, 0.32, 0.26, 0.28, 0.26, 0.34, 0.36, 0.32, 0.4, 0.36, 0.36, 0.38, 0.38, 0.36, 0.28, 0.28, 0.24, 0.28, 0.28, 0.26, 0.34, 0.36, 0.36, 0.4, 0.36, 0.36, 0.38, 0.38, 0.36, 0.28, 0.28, 0.24, 0.28, 0.28, 0.26, 0.34, 0.36, 0.38, 0.4, 0.36, 0.36, 0.38, 0.38, 0.36, 0.28, 0.26, 0.24, 0.3, 0.3, 0.28, 0.36, 0.4, 0.4, 0.4, 0.36, 0.36, 0.38, 0.36, 0.36, 0.28, 0.28, 0.24, 0.3, 0.32, 0.36, 0.36, 0.38, 0.4, 0.4, 0.36, 0.36, 0.38, 0.32, 0.34, 0.28, 0.24, 0.24, 0.28, 0.26, 0.32, 0.38, 0.38, 0.46, 0.36, 0.34, 0.36, 0.32, 0.24, 0.26, 0.24, 0.28, 0.3, 0.3, 0.3, 0.36, 0.36, 0.4, 0.4, 0.38, 0.36, 0.32, 0.24, 0.24, 0.3, 0.32, 0.38, 0.36, 0.32, 0.3, 0.36, 0.4, 0.44, 0.44, 0.34, 0.32, 0.3, 0.26, 0.28, 0.26, 0.26, 0.36, 0.3, 0.34, 0.36, 0.38, 0.42, 0.44, 0.42]]
minimum = min(gapdata[2])
indices = np.argsort(gapdata[2])[:5]
#indices = [i for i, v in enumerate(gapdata) if v == minimum]
print("indices",indices)
for i in indices:
    print("The best gap penalty is",gapdata[0][i])
    print("The best gap start penalty is",gapdata[1][i])
    print("The false positive rate is", gapdata[2][i])
"""
