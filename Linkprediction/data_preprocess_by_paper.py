#%%
data_folder='./data/'
graph_path='./graph/'
matrix_path='./matrix/'
tensor_path='./tensors/'
import networkx as nx
import pandas as pd
import numpy as np
import scipy.sparse
# %%
paper_num=24251
author_num=42614
# Data = open(data_folder+"paper_reference.csv", "r")
# next(Data, None)  # skip the first line in the input file
ref_table = pd.read_csv(data_folder+"paper_reference.csv",index_col=None,dtype=int)
#%% 单纯的引用关系
paper_ref=nx.DiGraph()
for i in range (paper_num):
    paper_ref.add_node(i)
for index, row in ref_table.iterrows():
    paper_ref.add_edge(row.paper_id,row.reference_id) #from referrer to referee
# paper_ref=nx.parse_edgelist(Data, delimiter=',', create_using=nx.DiGraph,
#                       nodetype=int)
# %%
nx.write_gexf(paper_ref,graph_path+"paper_reference_digraph.gexf")
#%% 引用关系加上同一个作者不同论文间有关系
paper_info_all = pd.read_csv(data_folder+"author_paper_all_with_year.csv",index_col=0,dtype=int)
ref_table = pd.read_csv(data_folder+"paper_reference.csv",index_col=None,dtype=int)
paper_ref_coauthor=nx.Graph()
for i in range(paper_num):
    paper_ref_coauthor.add_node(i)
for index, row in ref_table.iterrows():
    paper_ref_coauthor.add_edge(row.paper_id,row.reference_id) #from referrer to referee

for author in range(author_num):
    block=paper_info_all.loc[author]
    if block.ndim==1:
        continue
    papers=block.paper_id
    for paper_1 in papers:
        for paper_2 in papers:
            if (paper_1==paper_2):
                continue
            paper_ref_coauthor.add_edge(paper_1,paper_2)
nx.write_gexf(paper_ref_coauthor,graph_path+"paper_ref_coauthor_graph.gexf")
#%% 引用关系加上同一个作者不同论文间有关系，但是联系只有一条边的就算了
paper_info_all = pd.read_csv(data_folder+"author_paper_all_with_year.csv",index_col=0,dtype=int)
ref_table = pd.read_csv(data_folder+"paper_reference.csv",index_col=None,dtype=int)
paper_ref_coauthor=nx.MultiDiGraph()
for i in range(paper_num):
    paper_ref_coauthor.add_node(i)
for author in range(author_num):
    block=paper_info_all.loc[author]
    if block.ndim==1:
        continue
    papers=block.paper_id
    for paper_1 in papers:
        for paper_2 in papers:
            if (paper_1>=paper_2):
                continue
            paper_ref_coauthor.add_edge(paper_1,paper_2)
edge_array=np.array(list(paper_ref_coauthor.edges()))
paper_ref_coauthor_MultiDi=paper_ref_coauthor
paper_ref_coauthor=nx.Graph()
for i in range(paper_num):
    paper_ref_coauthor.add_node(i)
    
(_,paper_id_index)=np.unique(edge_array[:,0],return_index=True)
paper_id_index=paper_id_index[1:]
paper_id_index_before=0
for paper_id_index_next in paper_id_index:
    end_paper_id_before=-1
    edge_array[paper_id_index_before:paper_id_index_next,1]=np.sort(edge_array[paper_id_index_before:paper_id_index_next,1])
    for i in range(paper_id_index_before,paper_id_index_next):
        if (end_paper_id_before!=edge_array[i,1]):
            end_paper_id_before=edge_array[i,1]
        else:
            paper_ref_coauthor.add_edge(edge_array[i,0],edge_array[i,1])
    paper_id_index_before=paper_id_index_next
   
for index, row in ref_table.iterrows():
    paper_ref_coauthor.add_edge(row.paper_id,row.reference_id) #from referrer to referee
nx.write_gexf(paper_ref_coauthor,graph_path+"paper_ref_coauthor_graph_removed_weak_connection.gexf")
#%% 对图做embedding
from EasyN2V import EasyN2V
paper_ref_coauthor=nx.read_gexf(graph_path+"paper_ref_coauthor_graph.gexf")
n2v=EasyN2V(p=1,q=1,d=128,w=2)
n2v.fit(paper_ref_coauthor)
embeddings = []
for node in paper_ref_coauthor.nodes:
    embeddings.append(list(n2v.predict(node)))  
embeddings=np.array(embeddings)
np.save(matrix_path+'paper_ref_coauthor_N2V(1,1,128,2).npy',embeddings)
# %%
L=nx.laplacian_matrix(paper_ref.to_undirected(reciprocal=False, as_view=False))
scipy.sparse.save_npz(matrix_path+'Laplacian_sparse_'+'paper_reference.npz', L)
# %%
paper_info_labeled = pd.read_csv(data_folder+"labeled_papers_with_authors.csv",index_col=1)
paper_id=np.unique(paper_info_labeled.index)
# %%
paper_id_vector=np.zeros(paper_id.shape,dtype=int)
label_vector=np.zeros(paper_id.shape,dtype=int)
for i in range(paper_id.size):
    id=paper_id[i]
    block=paper_info_labeled.loc[id]
    if (block.ndim==1):
        label=block.label
    else:
        label=np.array(block.label)[0]
    paper_id_vector[i]=int(id)
    label_vector[i]=int(label)
np.save(matrix_path+'paper_id_vector.npy',paper_id_vector)
np.save(matrix_path+'label_vector.npy',label_vector)
# %% wqw1:大数据想要两个数据文件（最好存成txt），用torch.genfromtxt()读取，第一个读取后的ndarray 第一列是论文名字（结点） 第二列是论文对应的作者进行降维后的向量（先降到四五千维左右？）（结点特征）最后一列是标签（会议）

# feature_low_dimension=np.load(matrix_path+'feature_PCA.npy')
# feature_low_dimension=np.genfromtxt(matrix_path+"node2vec_from_paper_ref.txt",delimiter=',')
feature_low_dimension=np.load(matrix_path+"node2vec(paper_ref&author_multigraph)feature.npz")['arr_0']
feature_low_dimension=feature_low_dimension[42614:,:]
paper_info_all = pd.read_csv(data_folder+"author_paper_all_with_year.csv",index_col=1)
paper_id=np.unique(paper_info_all.index)
paper_info_labeled = pd.read_csv(data_folder+"labeled_papers_with_authors.csv",index_col=1)

(known_id,known_index)=np.unique(paper_info_labeled.index,return_index=True)
paper_label=np.full(paper_id.shape,-1)
for i in range(known_id.size):
    paper_label[known_id[i]]=paper_info_labeled.iloc[known_index[i]]['label']
result_matrix=np.concatenate((np.transpose([paper_id]),feature_low_dimension,np.transpose([paper_label])),axis=1)
np.savetxt(matrix_path+'id_lowDauthor_label_all.txt',result_matrix,delimiter=',')
#%% 第二个文件是论文的引用关系，也是用torch.genfromtxt()读取，读取后的ndarray，每一行有两个数据，第一列的论文先写，第二列的论文引用第一个编号的论文。
ref_table = pd.read_csv(data_folder+"paper_reference.csv",dtype=int)
d={'ref':ref_table.reference_id,'paper':ref_table.paper_id}
new_table=pd.DataFrame(d)
new_table.to_csv(matrix_path+'ref_to_paper_array.txt', header=False, index=False)
# %% paper_ref图加上node2vec(paper_ref)的结点特征
import networkx as nx
import EasyN2V 
n2v = EasyN2V(p = 1, q = 1, d = 1000,w=10)
n2v.fit(G)
embeddings = []
for node in G.nodes:
    embeddings.append(list(n2v.predict(node)))  
embeddings=np.array(embeddings)
# %% 
for node in G.nodes:
    G.nodes[node]['data']=embeddings[node]
nx.write_gpickle(G,graph_path+"paper_reference_digraph_with_node2vec_feature.gpickle")