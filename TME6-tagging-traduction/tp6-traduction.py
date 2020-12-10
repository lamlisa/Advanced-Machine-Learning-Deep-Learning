import logging
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

import time
import re
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
logging.basicConfig(level=logging.INFO)

FILE = "data/en-fra.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# --------------------------------- Utils -------------------------------------
# =============================================================================

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]


class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]


def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


class TradDatasetBPE():
    def __init__(self,data,spp,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor(spp.encode(orig, out_type=int)+[Vocabulary.EOS]),torch.tensor(spp.encode(dest, out_type=int)+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]


# =============================================================================
# ---------------------------- Encoder/Decoder --------------------------------
# =============================================================================
    
class Encoder(nn.Module):
    
    def __init__(self, in_size, latent_size, emb_size):
        
        # in_size : original vocabulary size
        
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(in_size, emb_size)
        self.rnn = nn.GRU(emb_size, latent_size)
        
    def forward(self, x, h):
        
        # x : length*batch_size
        # h : 1*batch_size*latent_size
        
        # output : length*batch_size*latent_size
        
        return self.rnn(self.embedding(x), h)[0]


class Decoder(nn.Module):
    
    def __init__(self, out_size, latent_size, emb_size):
        
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(out_size, emb_size)
        self.act = nn.ReLU()
        self.rnn = nn.GRU(emb_size, latent_size)
        self.linear = nn.Linear(latent_size, out_size)
    
    def forward(self, x, h):
        
        # x : batch_size
        # h : 1*batch_size*latent_size
        
        # output : 
        
        x = x.view(1,-1)
        x = self.embedding(x)
        x = self.act(x)
        output, h = self.rnn(x)
        return self.linear(h).squeeze(), h
    
    
# =============================================================================
# --------------------------- Data preparation --------------------------------
# =============================================================================
    
MAX_LEN = 15

# --- Load data
with open(FILE) as f:
    lines = f.readlines()

# --- train/test separation
lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

# --- avec vocabulaire de mots
vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

# --- avec Byte Pair Encoding
spp = spm.SentencePieceProcessor(model_file='model.model')
datatrain = TradDatasetBPE("".join(lines[:idxTrain]),spp,max_len=MAX_LEN)
datatest = TradDatasetBPE("".join(lines[idxTrain:]),spp,max_len=MAX_LEN)


# =============================================================================
# -------------------------------- Training -----------------------------------
# =============================================================================
    
# --- Parameters

in_size = len(vocEng)
out_size = len(vocFra)
latent_size = 64
emb_size1 = 32
emb_size2 = 32
batch_size = 128

nb_epoch = 20
lr = 0.01
p = 0.9999
p_decay = 0.9999


train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(datatest, collate_fn=collate, batch_size=batch_size, shuffle=True, drop_last=True)


encoder = Encoder(in_size, latent_size, emb_size1)
decoder = Decoder(out_size, latent_size, emb_size2)
h0 = torch.zeros(1, batch_size, latent_size)

loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=Vocabulary.PAD)
params = list(encoder.parameters()) + list(decoder.parameters())
optim = torch.optim.Adam(params, lr=lr)


writer = SummaryWriter()
debut = time.time()
cpt_train = 0
cpt_test = 0
for epoch in range(nb_epoch):
    
    print(f"-------------- TRAIN {epoch} ----------------")
    
    loss_sum = 0 
    correct = 0 # number of correct prediction in the epoch
    cpt_epoch = 0 # nombre d'éléments prédits durant l'epoch

    for x, len_x, y, len_y in train_loader:

        x = x.to(device)
        y = y.to(device)
        
        h = encoder(x, h0)
        y = torch.cat((torch.tensor([Vocabulary.SOS]*batch_size).view(1,-1), y), dim=0)
        
        loss = 0
        correct_batch = 0
        cpt_batch = 0
        if np.random.random() < p: # mode contraint

            for i in range(y.shape[0]-1):
                
                yhat, h = decoder(y[i], h)
                loss += loss_fn(yhat, y[i+1])
                    
                mask = y[i]!=Vocabulary.PAD
                correct_batch += (yhat[mask].argmax(dim=1)==y[i+1][mask]).sum()
                cpt_batch += mask.sum()
                
            loss_sum += loss.item()   
            loss /= cpt_batch
        
        else: # mode non contraint
            
            yhat, h = decoder(y[0], h)
            for i in range(1, y.shape[0]):
                
                logit = yhat.argmax(dim=1)
                yhat, h = decoder(logit, h)
                loss += loss_fn(yhat, y[i])
                
                mask = logit!=Vocabulary.PAD
                correct_batch += (yhat[mask].argmax(dim=1)==y[i][mask]).sum()
                cpt_batch += mask.sum()
                
            loss_sum += loss.item()
            loss /= cpt_batch
        
        accuracy_batch = correct_batch/float(cpt_batch)
        loss_sum += loss.item()
        correct += correct_batch
        cpt_epoch += cpt_batch

        writer.add_scalar("loss/batch/train", loss.item(), cpt_train)
        writer.add_scalar("accuracy/batch/train", accuracy_batch, cpt_train)
        
        if cpt_train%1==0:
            print(f"loss/batch : {loss.item()}")
            print(f"accuracy/batch : {accuracy_batch}\n")
            
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # augmenter proba du mode non contraint
        if p > 0.5:
            p *= p_decay
        
        cpt_train += 1
        
    writer.add_scalar("loss/epoch/train", loss_sum/float(cpt_epoch), epoch)
    writer.add_scalar("accuracy/epoch/train", correct/float(cpt_epoch), epoch)

    
    print(f"-------------- TEST {epoch} ----------------")
    
    with torch.no_grad():
        
        loss_sum = 0 
        correct = 0
        cpt_epoch = 0
        
        for x, len_x, y, len_y in test_loader:
        
            x = x.to(device)
            y = y.to(device)
            
            h = decoder(x, h0)
            y = torch.cat((torch.tensor([Vocabulary.SOS]*batch_size).view(1,-1), y), dim=0)
            
            loss = 0
            correct_batch = 0
            cpt_batch = 0
            for i in range(y.shape[0]-1):
                
                yhat, h = decoder(y[i], h)
                loss += loss_fn(yhat, y[i+1])
                    
                mask = y[i]!=Vocabulary.PAD
                correct_batch += (yhat[mask].argmax(dim=1)==y[i+1][mask]).sum()
                cpt_batch += mask.sum()
                
            loss_sum += loss.item()   
            loss /= cpt_batch
            
            accuracy_batch = correct_batch/float(cpt_batch)
            loss_sum += loss.item()
            correct += correct_batch
            cpt_epoch += cpt_batch
    
            writer.add_scalar("loss/batch/test", loss.item(), cpt_test)
            writer.add_scalar("accuracy/batch/test", accuracy_batch, cpt_test)
            
            if cpt_train%1==0:
                print(f"loss/batch : {loss.item()}")
                print(f"accuracy/batch : {accuracy_batch}\n")
                
            cpt_test +=1
    
        writer.add_scalar("loss/epoch/test", loss_sum/float(cpt_epoch), epoch)
        writer.add_scalar("accuracy/epoch/test", correct/float(cpt_epoch), epoch)
    
fin = time.time()
print(f"time : {(fin-debut)/60} min")
    
    
# =============================================================================
# ---------------------------- Test traduction --------------------------------
# =============================================================================

# --- avec vocabulaire de mots

with torch.no_grad():
    eng, fra = test_loader.dataset[44]
    h0 = torch.zeros(1,1,latent_size)
    ht = encoder(eng.view(-1,1), h0)
    tmp = torch.cat((torch.tensor([Vocabulary.SOS]).view(1,1), fra.view(-1,1)))
    res = []
    for i in range(len(tmp)-1):
        
        yhat, ht = decoder(tmp[i], ht)
        res.append(yhat.argmax().item())
    
    print(vocEng.getwords(eng))
    print(vocFra.getwords(fra))
    print(vocFra.getwords(res))

# --- avec Byte Pair Encoding
    
    
