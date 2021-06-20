from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import csv
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, one_hot_embedding
from model import Net

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')#300
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')#0.01
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')#5e-4
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')#16
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--Lambda', type=float, default=1,
                    help='Please refer to Eqn. (18) in original paper')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
device = torch.device('cuda:0')
# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()
pseudoOne = torch.ones((24525, 1), dtype=torch.float)
# adj = adj.to(device)
# Model and optimizer
model = Net(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            adj=adj,
            dropout_rate=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    pseudoOne = pseudoOne.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)


acc = 0
def train(epoch, acc):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train = loss_gcn
    loss_train.backward(retain_graph=True)
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if acc_val >= acc:
        torch.save(model.state_dict(),'DMtest.pkl')
        acc = acc_val
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch, acc)

# torch.save(model.state_dict(),'DMtest.pkl')

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

def test():
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
    preds = output.max(1)[1].type_as(labels)
    print("Test complete!")
    return preds

model.load_state_dict(torch.load('DMtest.pkl'))

# Testing

prediction = test().cpu()
result=(np.array(prediction)).tolist()
test_dict = {'predicted':result}
result = pd.DataFrame(test_dict, index = [0 for _ in range(len(result))])
result.to_csv("submission.csv", index = False, sep = ',')
