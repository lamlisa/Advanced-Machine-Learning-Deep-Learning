import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from utils import *
from generate import *
from textloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load data
with open ("trump_full_speech.txt","r") as f:
    text = f.read()


# =============================================================================
# --------------------------------- Training ----------------------------------
# =============================================================================
    
    
# --- Parameters
dico_size = len(id2lettre)
emb_size = 60
batch_size = 100
latent_size = 20

nb_epoch = 5
lr = 0.05


dataloader = DataLoader(TextDataset(text), collate_fn=collate_fn, batch_size=batch_size, shuffle=True, drop_last=True)

model = LSTM(dico_size, latent_size, emb_size).to(device)
h = torch.zeros(batch_size, latent_size).to(device)

loss_fn = nn.CrossEntropyLoss(reduction='none')
optim = torch.optim.Adam(model.parameters(), lr=lr)


writer = SummaryWriter()
debut = time.time()
cpt = 0
for epoch in range(nb_epoch):
    
    print(f"-------------- EPOCH {epoch} ----------------")
    
    loss_sum = 0 # pour calculer la loss/epoch
    for x in dataloader:

        x = x.to(device)
        latent = model.forward(x[:-1], h)
        
        loss = torch.zeros(x.shape[0]-1, batch_size)
        for i in range(x.shape[0]-1):
            yhat = model.decode(latent[i])
            loss[i] = loss_fn(yhat, x[i+1])
        loss = maskedCrossEntropy(loss, x[1:], PAD_IX)
        
        if cpt%100==0:
            print(f"{cpt} : {loss.item()}")   
        writer.add_scalar("loss/batch/train", loss.item(), cpt)
        loss_sum += loss
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        cpt += 1
        
    writer.add_scalar("loss/epoch/train", loss_sum/len(dataloader), epoch)
    
fin = time.time()
print(f"time : {(fin-debut)/60} min")
   

# =============================================================================
# ----------------------------- Test generation -------------------------------
# =============================================================================


start = "The world is..."

# --- Random
generate(model, model.embedding, model.decode, EOS_IX, type_gen='random', start=start, maxlen=200)

# --- Argmax
generate(model, model.embedding, model.decode, EOS_IX, type_gen='argmax', start=start, maxlen=200)

# --- Beam search
k = 3
generate_beam(model, model.embedding, model.decode, EOS_IX, k=k, start=start, maxlen=200)

# --- Nucleus Sampling
k = 3
generate_nucleus_sampling(model, model.embedding, model.decode, k=k, EOS_IX, start=start, maxlen=200)
