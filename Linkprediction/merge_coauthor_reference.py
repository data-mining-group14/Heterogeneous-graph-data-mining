#%%
data_folder='./data/'
graph_path='./graph/'
matrix_path='./matrix/'
import networkx as nx
import pandas as pd
import numpy as np
#%%
def merge(coauthorship:nx.MultiGraph,reference:nx.MultiDiGraph,empty_author)->nx.MultiGraph:
    merged=empty_author.copy()
    for start,end,data in coauthorship.edges(data=True):
        merged.add_edge(int(start),int(end),**data)
    for start,end,data in reference.edges(data=True):
        merged.add_edge(int(start),int(end),**data)
    return merged

#%%
# labeled_coauthor=nx.read_gexf(graph_path+"labeled_coauthor.gexf")
all_coauthor=nx.read_gexf(graph_path+"all_coauthor.gexf")
# labeled_reference=nx.read_gexf(graph_path+"refence_between_author_labeled.gexf")
all_reference=nx.read_gexf(graph_path+"refence_between_author_all.gexf")
# %%
empty_author=nx.MultiGraph()
empty_author.add_nodes_from(range(42613))
# labeled_coauthor_reference=merge(labeled_coauthor,labeled_reference,empty_author)
# nx.write_gexf(labeled_coauthor_reference,graph_path+"labeled_coauthor_reference.gexf")
all_coauthor_reference=merge(all_coauthor,all_reference,empty_author)
# nx.write_gexf(all_coauthor_reference,graph_path+"all_coauthor_reference.gexf")
nx.write_gpickle(all_coauthor_reference,graph_path+"all_coauthor_reference.gpickle")
# %%

# %%
#read in file from disk, skip repeated labor
# labeled_coauthor_reference=nx.read_gexf(graph_path+"labeled_coauthor_reference.gexf")
# all_coauthor_reference=nx.read_gpickle(graph_path+"all_coauthor_reference.gpickle")
# all_coauthor_reference_single_undi=nx.Graph(all_coauthor_reference)
# %%
# import EasyN2V
# n2v = EasyN2V(p = 1, q = 1, d = 100,w=10)
# n2v.fit(all_coauthor_reference_single_undi)
# embeddings = []
# for node in all_coauthor_reference_single_undi.nodes:
#     embeddings.append(list(n2v.predict(node)))  
# embeddings=np.array(embeddings)
# np.save(matrix_path+"all_coauthor_reference_single_undi(p = 1, q = 1, d = 100,w=10).npy",embeddings)
# %%
# embeddings=np.load(matrix_path+"all_coauthor_reference_Node2Vec(p = 1, q = 1, d = 1000,w=10).npy")
# %%
