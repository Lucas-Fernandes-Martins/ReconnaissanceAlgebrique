#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Tuple


# In[2]:


import torch
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch.utils.data import DataLoader, Subset
from sklearn.calibration import LabelEncoder
from generate_dataset import generate_dataset
import json 


# Constants

# In[3]:


GENERATED_DATASET_SIZE = 130
BATCH_SIZE = 32
TRAIN_SAMPLES = 50


# In[4]:


def dict_to_geometric_representation(in_graph_dict: dict, encoder) -> Data:
    node_list = []
    edge_mappings = []
    def traverse_graph(graph = in_graph_dict):
        nonlocal node_list
        nonlocal edge_mappings
        curr_node_index = len(node_list) - 1
        encoded_data = encoder(graph["val"])
        node_list.append(encoded_data)
        if hasattr(graph,"children"):
            for child in graph["children"]:
                edge_mappings.append((graph["id"], traverse_graph(child)) )
        return curr_node_index
    traverse_graph()
    nodes = torch.tensor(node_list,dtype=torch.float32)
    edges = torch.tensor([[x[0] for x in edge_mappings], [x[1] for x in edge_mappings]], dtype=torch.long) # Probably slow and mentally degenerated
    geom_data = Data(x=nodes, edge_index=edges)
    return geom_data


# In[5]:


OPERATIONS = ["ADD", "MUL", "FUNC", "POW"]
FUNCTIONS = ["SIN", "COS", "TAN", "EXP", "LOG", "f", "g", "h"]
ATOMICS = ["LITERAL", "VARIABLE"]
VARIABLE_ALPHABET = [chr(x) for x in range(ord("a"), ord("z")+1) if chr(x) not in ["f", "g", "h"]]


# In[6]:


def make_node_attribute_encoder(label_encoder:LabelEncoder, rep = 3):
    def node_attr_encoder(attr):
        if isinstance(attr, str):
            res = label_encoder.transform([attr])
            return [res[0]]*(rep + 1)
        else:
            return [0] + [attr]*rep
            
    return node_attr_encoder


# In[7]:


def create_dataset_class(expression):
    # Will it be the same for both datasets ? 
    le = LabelEncoder()
    le.fit(OPERATIONS+FUNCTIONS+ATOMICS+VARIABLE_ALPHABET)
    class MathExpressionDataset(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
            super().__init__(root, transform, pre_transform, pre_filter, force_reload=True)
            self.load(self.processed_paths[0])
            
        @property
        def raw_file_names(self):
            return ['math_datagen.json']

        @property
        def processed_file_names(self):
            return ['data.pt']
        

        def process(self):
            # Read data into huge `Data` list.
            data_list = []
            for file in self.raw_file_names:
                with open(file) as file_handle:
                    object_data = json.load(file_handle)
                    for comparison in object_data:
                        expr = comparison[expression]
                        score = comparison["score"]
                        geometric_expr = dict_to_geometric_representation(expr, make_node_attribute_encoder(le))
                        geometric_expr.y = score #torch.tensor([score],dtype=torch.float32)
                        data_list.append(geometric_expr)
                        
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]
            self.save(data_list, self.processed_paths[0])
    return MathExpressionDataset
    


# In[8]:


class ExpressionPairDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__()
        self.dataset_l = create_dataset_class("expr_l")(root+"_l",transform=None, pre_transform=None, pre_filter=None)
        self.dataset_r = create_dataset_class("expr_r")(root+"_r",transform=None, pre_transform=None, pre_filter=None)
        
    @property 
    def num_features(self):
        return self.dataset_l.num_features
    
    
    def __getitem__(self, idx):
        return self.dataset_l[idx], self.dataset_r[idx]


# In[9]:


generate_dataset(GENERATED_DATASET_SIZE,"math_datagen.json")
dataset = ExpressionPairDataset(root="/dataset")


# In[ ]:





# In[10]:


from torch import nn
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# In[11]:


class FormulaNet(nn.Module):
    def __init__(self, hidden_channels: int, embedding_space: int):
        super(FormulaNet, self).__init__()
        self.dense_1 = Linear(dataset.num_features, dataset.num_features) 
        self.relu_1 = ReLU()
        self.gconv_1 = GCNConv(dataset.num_features, hidden_channels)
        self.gconv_2 = GCNConv(hidden_channels, hidden_channels)
        self.gconv_3 = GCNConv(hidden_channels, hidden_channels)
        self.dense_2 = Linear(hidden_channels, embedding_space)
    
    def forward(self, data: Data):
        data.x = self.dense_1(data.x)
        data.x = self.relu_1(data.x)
        data.x = self.gconv_1(data.x, data.edge_index)
        data.x = self.relu_1(data.x)
        data.x = self.gconv_2(data.x, data.edge_index)
        data.x = self.relu_1(data.x)
        data.x = self.gconv_3(data.x, data.edge_index)
        data.x = self.relu_1(data.x)
        data.x = global_mean_pool(data.x, data.batch)
        data.x = F.dropout(data.x, p=0.3,training=self.training)
        data.x = self.dense_2(data.x)
        return data.x
    


# In[12]:


class SiameseFormulaNet(nn.Module):
    def __init__(self, hidden_channels, embedding_space):
        super(SiameseFormulaNet, self).__init__()
        self.formulanet = FormulaNet(hidden_channels, embedding_space)
        # self.fc = nn.Sequential(
        #     Linear(embedding_space*2, embedding_space),
        #     ReLU(inplace=True),
        #     Linear(embedding_space, 1)
        # )
        # self.sigmoid = nn.Sigmoid() # TODO: Only used it for testing purposes, everything is subject to change Okay
    

    def forward(self, expr_l, expr_r):
        embed_l = self.formulanet(expr_l)
        embed_l = embed_l.view(embed_l.size()[0], -1)
        embed_r = self.formulanet(expr_r)
        embed_r = embed_r.view(embed_r.size()[0], -1)
        
        # output = torch.cat((embed_l, embed_r), 1)
        
        # output = self.fc(output)
        # output = self.sigmoid(output)
        # return output
        return embed_l, embed_r
        
        
        


# In[ ]:





# In[13]:


train_dataset = Subset(dataset, list(range(TRAIN_SAMPLES)))
test_dataset = Subset(dataset, list(range(TRAIN_SAMPLES, -1)))


# In[14]:


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB
# NOTE: Type ignore only for collate_fn_t ... make sure it doesn't get in the way of correct typing for the dataset
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate) # type: ignore
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate) # type: ignore


# In[15]:


device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[16]:


model = SiameseFormulaNet(32,64).to(device)


# In[17]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[18]:


from tqdm import tqdm


# In[19]:


def contrastive_loss(embed_l, embed_r, labels, margin=1.0, threshold=0.5):
    euclidean_distance = F.pairwise_distance(embed_l, embed_r)
    labels = (labels > threshold).float()  # Convert labels to 0 or 1
    loss_contrastive = torch.mean((1 - labels) * torch.pow(euclidean_distance, 2) +
                                  labels * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive


# In[84]:


def train(epoch):
    model.train()
    epoch_loss = 0
    for batch_l, batch_r in tqdm(train_loader, desc=f'Epoch {epoch}'):
        batch_l, batch_r = batch_l.to(device), batch_r.to(device)
        # print(batch_l)
        # print(batch_r)
        optimizer.zero_grad()
        embed_l, embed_r = model(batch_l, batch_r)
        # print(embed_l)
        # print(embed_r)
        loss = contrastive_loss(embed_l, embed_r, batch_l.y)
        # print(loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {epoch_loss / len(train_loader)}')


# In[85]:


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_l, batch_r in test_loader:
            batch_l, batch_r = batch_l.to(device), batch_r.to(device)
            embed_l, embed_r = model(batch_l, batch_r)
            euclidean_distance = F.pairwise_distance(embed_l, embed_r)
            pred = (euclidean_distance < 0.5).float()  # Adjust the threshold as needed
            correct += (pred == batch_l.y).sum().item()
            test_loss += contrastive_loss(embed_l, embed_r, batch_l.y).item()
    test_loss /= len(test_loader)
    acc = correct / len(test_dataset)
    print(f'Test Loss: {test_loss}, Accuracy: {acc}')


# In[86]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# In[87]:


num_epochs = 10
for epoch in range(num_epochs):
    train(epoch)
    test()


# # Misc

# In[26]:


from torchviz import make_dot


# In[45]:


make_dot(y_f.mean(), params=dict(formulanet.named_parameters()))

