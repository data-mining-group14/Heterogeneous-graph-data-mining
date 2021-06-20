import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import numpy as np

def dataprepare():
    df = pd.read_csv('./data/author_paper_all_with_year.csv')
    data=np.array(df)
    author_num=np.max(data[:,0])+1
    paper_num=np.max(data[:,1])+1
    print("author_num:"+str(author_num))
    print("paper_num:"+str(paper_num))
    author_paper_pair=np.zeros((author_num,paper_num))
    author_cop_pair=[]
    for i in range(data.shape[0]):
        author_paper_pair[data[i,0],data[i,1]]=1
        for j in range(1,10000):
            if((i+j)<data.shape[0] and data[i,1]==data[i+j,1]):
                author_cop_pair.append([data[i,0],data[i+j,0]])
                author_cop_pair.append([data[i+j,0],data[i,0]])
            else:
                break
    author_paper_pair=author_paper_pair.T
    author_cop_pair=np.array(author_cop_pair).T
    author_feature=np.zeros((author_num,16))
    for i in range(author_num):
        n=i+1
        for j in range(16):
            if(n%2==1):
                author_feature[i,j]=1
                n=(n-1)/2
            else:
                author_feature[i,j]=0
                n=n/2
    df2=pd.read_csv("./data/paper_reference.csv")
    reference=np.array(df2).T
    r=np.zeros_like(reference)
    r[0,:]=reference[1,:]
    r[1,:]=reference[0,:]
    reference=np.concatenate((reference,r),axis=1) 
    df3=pd.read_csv("./data/labeled_papers_with_authors.csv",dtype=np.int32)
    data=np.array(df3)
    label_num=max(data[:,1])+1
    train_idx=np.array(range(label_num))
    train_label=np.zeros(label_num)
    
    for i in range(data.shape[0]):
        train_label[data[i,1]]=data[i,3]
    pred_idx=np.array(pd.read_csv("./data/authors_to_pred.csv"))
    print("author data preprocessing finished!")
    return author_paper_pair,normalize(author_feature),author_cop_pair,reference,train_idx,train_label,pred_idx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


