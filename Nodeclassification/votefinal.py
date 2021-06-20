# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 00:39:50 2021

@author: Apple
"""

import csv
import numpy as np
import pandas as pd


paperResult1 = []
with open('data/submission_paperauthorn2v1000_mulheaGAT_16_0.01.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult1.append(row['predicted'])
paperResult2 = []
with open('data/submission_paperauthorn2v1000_multiheadGAT_0.01_300.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult2.append(row['predicted'])

paperResult3 = []
with open('data/submission_ARMA_2_1_2_1.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult3.append(row['predicted'])
paperResult4 = []
with open('data/submission_ARMA_32.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult4.append(row['predicted'])
paperResult5 = []
with open('data/submissionDenseLinkGAT.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult5.append(row['predicted'])
        

        
paperResult6 = []
with open('data/ARMA.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult6.append(row['0'])


paperResult7 = []
with open('data/GAT_head=1.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult7.append(row['0'])


paperResult8 = []
with open('data/GAT_head=8.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult8.append(row['0'])


paperResult9 = []
with open('data/Cheb.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult9.append(row['0'])
        
paperResult10 = []
with open('data/GCN.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult10.append(row['0'])
        
paperResult11 = []
with open('data/submissionSparseLinkGATmultihead.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult11.append(row['predicted'])

paperResult = []
for i in range(len(paperResult1)):
    vote = np.zeros(10,dtype = int)

    vote[int(paperResult1[i])] += 1
    vote[int(paperResult2[i])] += 1
    vote[int(paperResult3[i])] += 1
    vote[int(paperResult4[i])] += 2
    vote[int(paperResult5[i])] += 2
    vote[int(float(paperResult6[i]))] += 2
    vote[int(float(paperResult7[i]))] += 1
    vote[int(float(paperResult8[i]))] += 2
    vote[int(float(paperResult9[i]))] += 1
    vote[int(float(paperResult10[i]))] += 2
    vote[int(float(paperResult11[i]))] += 2
    
    if max(vote) > min(vote):
        paperResult.append(str(np.argmax(vote)))
    else:
        paperResult.append(paperResult11[i])
        

authorPaper = []
with open('data/author_paper_all_with_year.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        authorPaper.append([row['author_id'], row['paper_id']])
        
authorPaperLabel = {}
with open('data/labeled_papers_with_authors.csv') as f2:    
    reader2 = csv.DictReader(f2)
    for row2 in reader2:
        authorPaperLabel[row2['paper_id']]=row2['label']


authorResult = {}

for i in range(len(authorPaper)):
    author = authorPaper[i][0]
    paper = int(authorPaper[i][1])
    if author in authorResult:
        if paper <= 4843:
            authorResult[author].append(authorPaperLabel[str(paper)])
        else:
            authorResult[author].append(paperResult[paper])
        
    else:
        if paper <= 4843:
            authorResult[author] = [authorPaperLabel[str(paper)]]
        else:
            authorResult[author] = [paperResult[paper]]
        
        


sub = []
authorPred = []
with open('data/authors_to_pred.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        authorPred.append(row['author_id'])
        
for j in range(len(authorPred)):
    authorSet = list(set(authorResult.get(authorPred[j])))
    authorSet.sort()
    sub.append([authorPred[j],' '.join(authorSet)])

result = pd.DataFrame(sub, columns=['author_id','labels'], index = [0 for _ in range(len(sub))])
result.to_csv("finalresult.csv", index = False, sep = ',')