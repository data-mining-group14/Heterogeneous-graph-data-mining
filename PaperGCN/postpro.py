# -*- coding: utf-8 -*-
"""
Created on Sat May 29 21:21:14 2021

@author: Apple
"""

import csv
import numpy as np
import pandas as pd
paperResult = []
with open('submissionSparseGCN.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        paperResult.append(row['predicted'])

authorPaper = []
with open('author_paper_all_with_year.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        authorPaper.append([row['author_id'], row['paper_id']])
        
authorPaperLabel = {}
with open('labeled_papers_with_authors.csv') as f2:    
    reader2 = csv.DictReader(f2)
    for row2 in reader2:
        authorPaperLabel[row2['paper_id']]=row2['label']

authorResult = {}

for i in range(len(authorPaper)):
    author = authorPaper[i][0]
    paper = int(authorPaper[i][1])
    if author in authorResult:
        if paper <= 4843:
            authorResult[author].append(str(int(authorPaperLabel[str(paper)])))
        else:
            authorResult[author].append(str(int(float(paperResult[paper]))))
        
    else:
        if paper <= 4843:
            authorResult[author] = [str(int(authorPaperLabel[str(paper)]))]
        else:
            authorResult[author] = [str(int(float(paperResult[paper])))]
        
        


sub = []
authorPred = []
with open('authors_to_pred.csv') as f:    
    reader = csv.DictReader(f)
    for row in reader:
        authorPred.append(row['author_id'])
        
for j in range(len(authorPred)):
    authorSet = list(set(authorResult.get(authorPred[j])))
    authorSet.sort()
    sub.append([authorPred[j],' '.join(authorSet)])

result = pd.DataFrame(sub, index = [0 for _ in range(len(sub))])
result.to_csv("finalresultSparseGCN.csv", index = False, sep = ',')