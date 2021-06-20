# %%
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sklearn.linear_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from stellargraph.data import EdgeSplitter
import numpy as np
import pandas as pd
import networkx as nx
data_folder = './data/'
graph_path = './graph/'
matrix_path = './matrix/'
# result_path = './results/'
result_path = './vote/'
# %% load author pairs to predict
df = pd.read_csv(data_folder+'author_pairs_to_pred_with_index.csv', index_col=None)
new = df["author_pair"].str.split(" ", n=1, expand=True)
df['first author'] = new[0]  # .astype(np.int64)
df['second author'] = new[1]  # .astype(np.int64)
del df['author_pair']
author_pair_to_predict = np.stack((df['first author'], df['second author']), axis=1)
id = df['id']
# %%
all_coauthor_reference = nx.read_gpickle(graph_path+"all_coauthor_reference.gpickle")
# all_coauthor=nx.read_gexf(graph_path+'all_coauthor.gexf')
# all_reference=nx.read_gexf(graph_path+'refence_between_author_all.gexf')
#%%
graph_to_export_edge=all_coauthor_reference#.to_undirected()
graph_to_embed=all_coauthor_reference#.to_undirected()
# %%
edge_splitter = EdgeSplitter(graph_to_export_edge)
graph_train, pairs, labels = edge_splitter.train_test_split(p=0.1, method="global")
(pairs_train,pairs_test,labels_train,labels_test) = train_test_split(pairs, labels, train_size=0.75, test_size=0.25)
# %%
from EasyN2V import EasyN2V
class create_embedding:
    def __init__(self,G,p=1,q=1,d=100,w=4):
        n2v=EasyN2V(p,q,d,w,batch_words=256)
        n2v.fit(G)
        self.n2v=n2v
    def predict(self, u):
        return self.n2v.predict(u)
# %%
def author_pair2features(pairs, embed):
    return [embed(src)*embed(dst) for src, dst in pairs]

def train(link_train, link_labels, get_embedding):
    lr_clf = sklearn.linear_model.LogisticRegressionCV(scoring="roc_auc",max_iter=3000)
    clf = Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])
    link_features = author_pair2features(link_train, get_embedding)
    clf.fit(link_features, link_labels)
    return clf
#%%
p=1;q=1;d=100;w=4;w_list=range(2,10)
#%%
def different_w(w):
    embedding_train=create_embedding(graph_to_embed,w=w).predict
    clf = train(pairs_train, labels_train, embedding_train)
    link_features = author_pair2features(author_pair_to_predict, embedding_train)
    predicted = clf.predict_proba(link_features)[:, 1]
    result = pd.DataFrame(np.stack((id, predicted), axis=1),columns=['id', 'label'], index=id)
    result['id'] = result['id'].astype(int)
    result_file_name='all_coauthor_reference_p=0.1_train=0.75_Node2Vec(p={},q={},d={},w={})_Harmard_LR.csv'.format(p,q,d,w)
    result.to_csv(result_path+result_file_name, index=False)

def different_d(d):
    embedding_train=create_embedding(graph_to_embed,d=d).predict
    clf = train(pairs_train, labels_train, embedding_train)
    link_features = author_pair2features(author_pair_to_predict, embedding_train)
    predicted = clf.predict_proba(link_features)[:, 1]
    result = pd.DataFrame(np.stack((id, predicted), axis=1),columns=['id', 'label'], index=id)
    result['id'] = result['id'].astype(int)
    result_file_name='all_coauthor_reference_p=0.1_train=0.75_Node2Vec(p={},q={},d={},w={})_Harmard_LR.csv'.format(p,q,d,w)
    result.to_csv(result_path+result_file_name, index=False)

def different_p(p):
    embedding_train=create_embedding(graph_to_embed,p=p).predict
    clf = train(pairs_train, labels_train, embedding_train)
    link_features = author_pair2features(author_pair_to_predict, embedding_train)
    predicted = clf.predict_proba(link_features)[:, 1]
    result = pd.DataFrame(np.stack((id, predicted), axis=1),columns=['id', 'label'], index=id)
    result['id'] = result['id'].astype(int)
    result_file_name='all_coauthor_reference_p=0.1_train=0.75_Node2Vec(p={},q={},d={},w={})_Harmard_LR.csv'.format(p,q,d,w)
    result.to_csv(result_path+result_file_name, index=False)

def different_q(q):
    embedding_train=create_embedding(graph_to_embed,q=q).predict
    clf = train(pairs_train, labels_train, embedding_train)
    link_features = author_pair2features(author_pair_to_predict, embedding_train)
    predicted = clf.predict_proba(link_features)[:, 1]
    result = pd.DataFrame(np.stack((id, predicted), axis=1),columns=['id', 'label'], index=id)
    result['id'] = result['id'].astype(int)
    result_file_name='all_coauthor_reference_p=0.1_train=0.75_Node2Vec(p={},q={},d={},w={})_Harmard_LR.csv'.format(p,q,d,w)
    result.to_csv(result_path+result_file_name, index=False)

import concurrent.futures
executor = concurrent.futures.ProcessPoolExecutor(max_workers=7)
_=executor.submit(different_d,80)
_=executor.submit(different_d,120)
_=executor.submit(different_p,1.2)
_=executor.submit(different_q,0.8)
executor.map(different_w,w_list)
executor.shutdown(wait=True,cancel_futures=False)