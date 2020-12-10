import itertools
import logging
from tqdm import tqdm

import os
import matplotlib.pyplot as plt
from datamaestro import prepare_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch
from typing import List
import time
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = prepare_dataset('org.universaldependencies.french.gsd')

# Format de sortie décrit dans
# https://pypi.org/project/conllu/


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
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
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

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
#dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)

logging.info("Vocabulary size: %d", len(words))


class Seq2Seq(nn.Module):
    
    def __init__(self, in_size, out_size, latent_size, emb_size):
        
        super(Seq2Seq, self).__init__()
        
        self.embedding = nn.Embedding(in_size, emb_size)
        self.encoder = nn.GRU(emb_size, latent_size)
        
        self.decoder = nn.Linear(latent_size, out_size)
        
    def forward(self, x):
        
        # x : length*batch_size
        # output : batch_size*latent_size
        
        return self.encoder(self.embedding(x))[0]
    
    def decode(self, h):
        
        # h : batch_size*latent_size
        # output : batch_size*out_size
        
        return self.decoder(h)
    
    
in_size = len(words)
out_size = len(tags)
latent_size = 20
emb_size = 1000
batch_size = 100

nb_epoch = 5
lr = 0.05

train_loader = DataLoader(train_data, collate_fn=collate, batch_size=batch_size, shuffle=True)
#dev_loader = DataLoader(dev_data, collate_fn=collate, batch_size=batch_size)
test_loader = DataLoader(test_data, collate_fn=collate, batch_size=batch_size)

model = Seq2Seq(in_size, out_size, latent_size, emb_size).to(device)
loss_fn = nn.CrossEntropyLoss(reduction='sum', ignore_index=Vocabulary.PAD)
optim = torch.optim.Adam(model.parameters(), lr=lr)

list_loss_train = []
list_loss_avg_train = []
list_acc_avg_train = []
list_loss_test = []
list_loss_avg_test = []
list_acc_avg_test = []

debut = time.time()
cpt = 0
for epoch in range(nb_epoch):
    
    print(f"-------------- TRAIN {epoch} ----------------")
    loss_avg = 0 # mean loss/batch

    acc_total = 0
    cpt_total = 0
    
    cpt = 0
    for x,y in train_loader:

        x = x.to(device)
        latent = model.forward(x)
        
        loss = 0
        acc_b = 0
        cpt_b = 0
        for i in range(x.shape[0]):
            yhat = model.decode(latent[i])
            mask = y[i]!=Vocabulary.PAD
            acc_b += (yhat[mask].argmax(dim=1)==y[i][mask]).sum()
            cpt_b += mask.sum()
            loss += loss_fn(yhat, y[i])
        loss /= cpt_b
        
        list_loss_train.append(loss.item())
        loss_avg += loss.item()
        acc_total += acc_b
        cpt_total += cpt_b
        
        if cpt%25==0:
            print(f"loss/batch : {loss.item()}")
            print(f"accuracy/batch : {torch.true_divide(acc_b,cpt_b)}\n")
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        cpt += 1
        
    list_loss_avg_train.append(torch.true_divide(loss_avg,len(train_loader)))
    list_acc_avg_train.append(torch.true_divide(acc_total,cpt_total))
    print(f"avg loss/batch : {list_loss_avg_train[-1]}")
    print(f"avg accuracy/batch : {list_acc_avg_train[-1]}\n")
    
    print(f"-------------- TEST {epoch} ----------------")
    with torch.no_grad():
        
        loss_avg = 0 # mean loss/batch
        acc_total = 0
        cpt_total = 0
        
        for x,y in test_loader:
        
            x = x.to(device)
            latent = model.forward(x)
            
            loss = 0
            acc_b = 0
            cpt_b = 0
            for i in range(x.shape[0]):
                yhat = model.decode(latent[i])
                mask = y[i]!=Vocabulary.PAD
                acc_b += (yhat[mask].argmax(dim=1)==y[i][mask]).sum()
                cpt_b += mask.sum()
                loss += loss_fn(yhat, y[i])
            loss /= cpt_b
            
            list_loss_test.append(loss.item())
            loss_avg += loss.item()
            acc_total += acc_b
            cpt_total += cpt_b
    
        list_loss_avg_test.append(torch.true_divide(loss_avg,len(test_loader)))
        list_acc_avg_test.append(torch.true_divide(acc_total,cpt_total))
        
        print(f"avg loss/batch : {list_loss_avg_test[-1]}")
        print(f"avg accuracy/batch : {list_acc_avg_test[-1]}\n")
    
fin = time.time()
print(f"time : {(fin-debut)/60} min")


path = f"Img/tag/e{emb_size}_b{batch_size}_lat{latent_size}_nb{nb_epoch}_lr{lr}/"
if not os.path.isdir(path):
    os.makedirs(path)

plt.figure()
plt.plot(list_loss_train)
plt.title(f"loss/batch train - {nb_epoch} epochs")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.savefig(path+"loss_batch_train.png")

plt.figure()
plt.plot(list_loss_test)
plt.title(f"loss/batch test - {nb_epoch} epochs")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.savefig(path+"loss_batch_test.png")

plt.figure()
plt.plot(list_loss_avg_train)
plt.plot(list_loss_avg_test)
plt.legend(["train","test"])
plt.title(f"mean loss/epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(path+"mean_loss_epoch.png")

plt.figure()
plt.plot(list_acc_avg_train)
plt.plot(list_acc_avg_test)
plt.legend(["train","test"])
plt.title(f"mean accuracy/epoch")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig(path+"mean_accuracy_epoch.png")
