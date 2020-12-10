import torch
import torch.nn as nn
import numpy as np
import logging
import csv
from torch.utils.data import Dataset
import string
import unicodedata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


# =============================================================================
# ------------------------------------ Utils ----------------------------------
# =============================================================================


LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' # NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def fill_na(mat):
    ix,iy = np.where(np.isnan(mat))
    for i,j in zip(ix,iy):
        if np.isnan(mat[i+1,j]):
            mat[i,j]=mat[i-1,j]
        else:
            mat[i,j]=(mat[i-1,j]+mat[i+1,j])/2.
    return mat


def read_temps(path):
    """Lit le fichier de températures"""
    data = []
    with open(path, "rt") as fp:
        reader = csv.reader(fp, delimiter=',')
        next(reader)
        for row in reader:
            data.append([float(x) if x != "" else float('nan') for x in row[1:]])
    return torch.tensor(fill_na(np.array(data)), dtype=torch.float)


def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


# =============================================================================
# ------------------------------------ RNN ------------------------------------
# =============================================================================
    

class RNN(nn.Module):
    
    def __init__(self,dim,out_size,latent_size):
        super(RNN, self).__init__()
        
        self.act1 = nn.Tanh()
        self.linear_x = nn.Linear(dim,latent_size,bias=False)
        self.linear_h = nn.Linear(latent_size,latent_size)
        
        self.linear_y = nn.Linear(latent_size,out_size)
        
    def one_step(self,x_t,h):
        
        # x_t: batch*dim
        # h: batch*latent
        
        return self.act1(self.linear_x(x_t)+self.linear_h(h)) # batch*latent
        
    def forward(self,x,h):
        
        # x: length*batch*dim
        # h: batch*latent
        
        l = []
        for t in range(x.shape[0]):
            h_new = self.one_step(x[t],h) # batch*latent
            l.append(h_new)
            h = h_new
        return torch.stack(l) # length*batch*latent
    
    def decode(self,h):
        return self.linear_y(h) # batch*output_size


# =============================================================================
# ---------------------------------- Datasets ---------------------------------
# =============================================================================
        
    
class TempClassificationDataset(Dataset):
    
    def __init__(self, X, length, mean, std):
        
        # X : tensor de taille nb_temp*nb_villes
        # self.X : tensor de taille nb_seq*length
        # self.y : tensor de taille nb_seq
        
        self.X = (X-mean)/std
        nb_temp, nb_villes = self.X.shape
        self.X = torch.stack([self.X[i:i+length].transpose(0,1) for i in range(nb_temp-length+1)], dim=0)
        self.X = self.X.view(-1,length)
        
        self.y = (torch.ones(nb_temp-length+1, nb_villes)*torch.arange(nb_villes)).view(-1)
        self.y = self.y.long()
        
    def __getitem__(self,index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)
    

class TempPredictionDataset(Dataset):
    
    def __init__(self, X, length, mean, std):
        self.X = (X-mean)/std
        self.length = length
        
    def __getitem__(self,index):   
        return self.X[index:index+self.length], self.X[index+1:index+1+self.length]
    
    def __len__(self):
        return len(self.X)-self.length
    

class SpeechDataset(Dataset):
    
    def __init__(self, X, y, length):
        self.X = X
        self.y = y
        self.length = length
        
    def __getitem__(self, index):
        return self.X[index:index+self.length], self.y[index+1:index+1+self.length]
    
    def __len__(self):
        return len(self.X)-self.length
