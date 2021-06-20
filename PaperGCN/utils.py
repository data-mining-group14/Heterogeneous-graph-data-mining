import numpy as np
import scipy.sparse as sp
import torch

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    labels_onehot = np.delete(labels_onehot, 10, 1)
    return labels_onehot


def load_data(path="../data/DM/", dataset="DM"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}id_lowDauthor_label_all.txt".format(path), delimiter=',')#("{}{}.content".format(path, dataset),dtype=np.dtype(str))
    features = np.load('{}paper_ref_coauthor_N2V(1,1,128,3).npy'.format(path))
    features = sp.csr_matrix(features, dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.load("{}paper_ref_coauthor_graph_removed_weak_connection.npy".format(path))
    adj = (torch.LongTensor(edges_unordered)).t()

    idx_train = range(4500)
    idx_val = range(4500, 4843)
    idx_test = range(4843, 24250)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

