import collections
import numpy as np
import pandas as pd
import itertools

import matplotlib
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import GCNConv

import networkx as nx

df_raw = pd.read_csv('/scratch-shared/martin/003_SPECS1K_ML/001_data/Specs935_ImageMeans_AfterQC_AnnotatedWithMOA.csv', sep=';')
df_raw.rename(columns={'Compound ID':'Compound_ID'}, inplace = True)
df = df_raw.copy()
df.dropna(subset=['Compound_ID'], inplace=True)
df = df[df['selected_mechanism'].str.contains('dmso')==False] # actually not dropping anything, since dropna already drop all dmso
df.drop(["Plate", "Plate_Well", "batch_id", "pertType", "Batch nr", "Compound_ID", "PlateID", "Well"], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["selected_mechanism"] = le.fit_transform(df.iloc[:,-1])


def tabular2graph(df, num_graphs=1000, num_nodes=500):
    #https://colab.research.google.com/drive/1_eR7DXBF3V4EwH946dDPOxeclDBeKNMD?usp=sharing
    data_lst = []
    
    for i in range(num_graphs):
        df_sub = df.sample(n=num_nodes)
        df_sub.reset_index(drop=True, inplace=True)
        teams = df_sub["Count_nuclei"].unique()
        all_edges = np.array([], dtype=np.int32).reshape((0, 2))

        for team in teams:
            team_df = df_sub[df_sub["Count_nuclei"] == team]
            players = team_df.index
            # Build all combinations, as all players are connected
            permutations = list(itertools.combinations(players, 2))
            edges_source = [e[0] for e in permutations]
            edges_target = [e[1] for e in permutations]
            team_edges = np.column_stack([edges_source, edges_target])
            all_edges = np.vstack([all_edges, team_edges])
        # Convert to Pytorch Geometric format
        edge_index = all_edges.transpose()
    
        node_features = df_sub.iloc[:,1:-2]
        labels = df_sub.iloc[:,-1]
    
        x = torch.tensor(node_features.values)    # node features
        y = torch.tensor(labels.values) # label in label endcoder form
        data = Data(x=x, edge_index=edge_index, y=y) # making graph in PyG
    
        data_lst.append(data)
    return data_lst

val_pct = 0.2

train_df = df.sample(frac = 1-val_pct)
valid_df = df.drop(train_df.index)

train_lst = tabular2graph(train_df, num_graphs=1000, num_nodes=300)
valid_lst = tabular2graph(valid_df, num_graphs=1000, num_nodes=300)

from torch_geometric.loader import DataLoader

trainloader = DataLoader(train_lst, batch_size=64)
validloader = DataLoader(valid_lst, batch_size=64)


from torch_geometric.nn import GCNConv, Sequential, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2129, 128, aggr='add')
        self.conv2 = GCNConv(128, 128, aggr='add')
        self.conv3 = GCNConv(128, 30, aggr='add')
        self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 30)
        )
             
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.fc(x)
        return x
    
    
device = "cuda" if torch.cuda.is_available() else "cpu"

#device = "cpu"

model = GCN().to(device)
n_epochs = 3000
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # weight_decay is for L2 regularization

best_acc = -1

for epoch in range(n_epochs):
    #----Training----#
    model.train()
    train_loss = []
    train_accs = []
    for batch in trainloader:
        batch.to(device)
        x, edge_index = batch.x, batch.edge_index
        logits = model(x.float().to(device), torch.tensor(edge_index[0]).long().to(device))
        loss = criterion(logits, torch.tensor(batch.y))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        acc = (logits.argmax(dim=-1) == batch.y).float().mean()

        train_loss.append(loss.item())
        train_accs.append(acc)
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    
    #----Validation----#
    model.eval()
    valid_loss = []
    valid_accs = []
    
    for batch in validloader:
        batch.to(device)
        x, edge_index = batch.x, batch.edge_index
        with torch.no_grad():
            logits = model(x.float().to(device), torch.tensor(edge_index[0]).long().to(device))
        loss = criterion(logits, torch.tensor(batch.y))
        acc = (logits.argmax(dim=-1) == batch.y).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    
    # Print the information.
    if epoch%20==0:
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
    if valid_acc > best_acc:
        torch.save(model.state_dict(), './model.ckpt')
        print(f'model saved at {epoch} epochs with acc {valid_acc}')
        best_acc = valid_acc