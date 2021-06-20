#%%
data_folder='./data/'
graph_path='./graph/'
import networkx as nx
import pandas as pd
import numpy as np
# %% 
paper_info_labeled = pd.read_csv(data_folder+"labeled_papers_with_authors.csv",index_col=1,dtype=int)
paper_id=np.unique(paper_info_labeled.index)
labeled_coauthor = nx.MultiGraph()
for paper in paper_id:
    block=paper_info_labeled.loc[paper]
    authors=block.author_id
    if (block.ndim==1):
        year=np.array(block.year)
        label=np.array(block.label)
        continue
    else:
        year=np.array(block.year)[0]
        label=np.array(block.label)[0]
    for author_1 in authors:
        for author_2 in authors:
            if (author_1==author_2):
                continue
            labeled_coauthor.add_edge(author_1,author_2,year=year,label=label)
nx.write_gexf(labeled_coauthor,graph_path+"labeled_coauthor.gexf")
# %%
paper_info_all = pd.read_csv(data_folder+"author_paper_all_with_year.csv",index_col=1,dtype=int)
paper_id=np.unique(paper_info_all.index)
all_coauthor = nx.MultiGraph()
for paper in paper_id:
    block=paper_info_all.loc[paper]
    authors=block.author_id
    if (block.ndim==1):
        year=np.array(block.year)
        continue
    else:
        year=np.array(block.year)[0]
    for author_1 in authors:
        for author_2 in authors:
            if (author_1==author_2):
                continue
            all_coauthor.add_edge(author_1,author_2,year=year,label=-1)
nx.write_gexf(all_coauthor,graph_path+"all_coauthor.gexf")
# %%
Data = open(data_folder+"paper_reference.csv", "r")
next(Data, None)  # skip the first line in the input file
paper_ref=nx.parse_edgelist(Data, delimiter=',', create_using=nx.DiGraph,
                      nodetype=int)
paper_ref_df = pd.read_csv(data_folder+"paper_reference.csv",index_col=0,dtype=int)

refence_between_author_labeled=nx.MultiDiGraph()
known_paper_id=np.unique(paper_info_labeled.index)
paper_id=np.unique(paper_ref_df.index)
for paper in paper_id:
    block=paper_ref_df.loc[paper]
    if block.ndim==1:
        reference_id=[block.reference_id]
    else:
        reference_id=block.reference_id
    if (not np.isin(paper,known_paper_id)):
        continue
    block=paper_info_labeled.loc[paper]
    if block.ndim==1:
        authors=[block.author_id]
        year=block.year
        label=block.label
    else:
        authors=block.author_id
        year=block.year.iloc[0]
        label=block.label.iloc[0]
    
    for reference_paper in reference_id:
        block=paper_info_all.loc[reference_paper]
        if block.ndim==1:
            ref_authors=[block.author_id]
        else:
            ref_authors=block.author_id
        for author in authors:
            for ref_author in ref_authors:
                if (author!=ref_author):
                    refence_between_author_labeled.add_edge(author,ref_author,year=year,label=label)
nx.write_gexf(refence_between_author_labeled,graph_path+"refence_between_author_labeled.gexf")
#%%
refence_between_author_all=nx.MultiDiGraph()
paper_id=np.unique(paper_ref_df.index)
for paper in paper_id:
    block=paper_ref_df.loc[paper]
    if block.ndim==1:
        reference_id=[block.reference_id]
    else:
        reference_id=block.reference_id
    block=paper_info_all.loc[paper]
    if block.ndim==1:
        authors=[block.author_id]
        year=block.year
    else:
        authors=block.author_id
        year=block.year.iloc[0]
    
    for reference_paper in reference_id:
        block=paper_info_all.loc[reference_paper]
        if block.ndim==1:
            ref_authors=[block.author_id]
        else:
            ref_authors=block.author_id
        for author in authors:
            for ref_author in ref_authors:
                if (author!=ref_author):
                    refence_between_author_all.add_edge(author,ref_author,year=year,label=-1)
nx.write_gexf(refence_between_author_all,graph_path+"refence_between_author_all.gexf")

