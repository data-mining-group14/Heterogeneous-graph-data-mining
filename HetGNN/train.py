from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy,dataprepare,normalize
from models import ClassifyNet
import pandas as pd
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_index', '-gi', type=int, default=0)


parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)

mask,feature,adj_author,adj_paper,train_idx,train_label,pred_idx= dataprepare()
feature=np.load('./data/featurew=3.npy')
mask1= torch.from_numpy(normalize(mask)).float().to(device)

model =ClassifyNet(feature_dim=feature.shape[1],
            hid_dim=args.hidden,
            class_num=10,
            dropout=args.dropout,
            mask1=mask1,
            mask2=mask1,
            device=device)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.9)
model=model.float().to(device)
feature= torch.from_numpy(feature).float().to(device)
adj_author = torch.from_numpy(adj_author).to(device)
adj_paper = torch.from_numpy(adj_paper).to(device)
train_idx = torch.from_numpy(train_idx).to(device)
train_label = torch.from_numpy(train_label).to(device)
pred_idx = torch.from_numpy(pred_idx).to(device)
    
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(feature, adj_author,adj_paper,mask1,mask1)
    loss_train = F.cross_entropy(output[train_idx], train_label[train_idx].long())
    acc_train = accuracy(output[train_idx], train_label[train_idx].long())
    loss_train.backward()
    optimizer.step()
    scheduler.step()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(feature, adj_author,adj_paper,mask1,mask1)
    preds = output.max(1)[1].type_as(train_label).cpu().numpy()
    result = pd.DataFrame(preds)
    result.to_csv("./result/ARMA.csv")
    
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
test()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))




