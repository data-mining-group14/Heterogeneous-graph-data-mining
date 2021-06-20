#%%
import numba
import numpy as np
import pandas as pd
import time
import warnings
import nodevectors
# Gensim triggers automatic useless warnings for windows users...
warnings.simplefilter("ignore", category=UserWarning)
import gensim
warnings.simplefilter("default", category=UserWarning)


import csrgraph as cg
from nodevectors.embedders import BaseNodeEmbedder

class Node2Vec_new(nodevectors.Node2Vec):
    def fit(self, G):
        """
        NOTE: Currently only support str or int as node name for graph
        Parameters
        ----------
        G : graph data
            Graph to embed
            Can be any graph type that's supported by csrgraph library
            (NetworkX, numpy 2d array, scipy CSR matrix, CSR matrix components)
        """

        if not isinstance(G, cg.csrgraph):
            G = cg.csrgraph(G, threads=self.threads)
        if G.threads != self.threads:
            G.set_threads(self.threads)
        # Because networkx graphs are actually iterables of their nodes
        #   we do list(G) to avoid networkx 1.X vs 2.X errors
        node_names = G.names
        if type(node_names[0]) not in [int, str, np.int32, np.uint32, 
                                       np.int64, np.uint64]:
            raise ValueError("Graph node names must be int or str!")
        # Adjacency matrix
        walks_t = time.time()
        if self.verbose:
            print("Making walks...", end=" ")
        self.walks = G.random_walks(walklen=self.walklen, 
                                    epochs=self.epochs,
                                    return_weight=self.return_weight,
                                    neighbor_weight=self.neighbor_weight)
        if self.verbose:
            print(f"Done, T={time.time() - walks_t:.2f}")
            print("Mapping Walk Names...", end=" ")
        map_t = time.time()
        self.walks = pd.DataFrame(self.walks)
        # Map nodeId -> node name
        node_dict = dict(zip(np.arange(len(node_names)), node_names))
        for col in self.walks.columns:
            self.walks[col] = self.walks[col].map(node_dict).astype(str)
        # Somehow gensim only trains on this list iterator
        # it silently mistrains on array input
        self.walks = [list(x) for x in self.walks.itertuples(False, None)]
        if self.verbose:
            print(f"Done, T={time.time() - map_t:.2f}")
            print("Training W2V...", end=" ")
            if gensim.models.word2vec.FAST_VERSION < 1:
                print("WARNING: gensim word2vec version is unoptimized"
                    "Try version 3.6 if on windows, versions 3.7 "
                    "and 3.8 have had issues")
        w2v_t = time.time()
        # Train gensim word2vec model on random walks
        self.model = gensim.models.Word2Vec(
            sentences=self.walks,
            vector_size=self.n_components,
            **self.w2vparams)
        if not self.keep_walks:
            del self.walks
        if self.verbose:
            print(f"Done, T={time.time() - w2v_t:.2f}")

class EasyN2V(Node2Vec_new):
    def __init__(self, p = 1, q = 1,d = 32, w = 10,batch_words=128):
            super().__init__(
                        n_components = d,
                        walklen = w,
                        epochs = 50,
                        return_weight = 1.0 / p,
                        neighbor_weight = 1.0 / q,
                        threads = 0,
                        w2vparams = {'window': 4,
                                    'negative': 5, 
                                    'epochs': 10,
                                    'ns_exponent': 0.5,
                                    'batch_words': batch_words,})
#%%
if __name__=='__main__':
    import networkx as nx
    import pandas as pd
    toy_barbell = nx.barbell_graph(7, 2)
    nx.draw_kamada_kawai(toy_barbell)
    n2v = EasyN2V(p = 1, q = 1, d = 2,w=10)
    n2v.fit(toy_barbell)
    embeddings = []
    for node in toy_barbell.nodes:
        embeddings.append(list(n2v.predict(node)))   
    # Construct a pandas dataframe with the 2D embeddings from node2vec.
    # We can easily divide the nodes into two clusters, and the groudtruth is denoted by distinct colors.
    toy_colors = ['red'] * 8 + ['blue'] * 8
    df = pd.DataFrame(embeddings, columns = ['x', 'y']) # Create pandas dataframe from the list of node embeddings
    df.plot.scatter(x = 'x', y = 'y', c = toy_colors)