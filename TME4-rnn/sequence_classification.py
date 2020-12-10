from utils import read_temps, RNN, device, TempClassificationDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
    

# --- Load data
  
temp_train = read_temps(r"data/tempAMAL_train.csv")
temp_test = read_temps(r"data/tempAMAL_test.csv")


# --- Parameters

nb_villes = 10
length = 100
batch_size = 64
latent_size = 20
lr = 0.001
nb_epoch = 1


# --- DataLoader

xtrain = temp_train[:,:nb_villes]
xtest = temp_test[:,:nb_villes]
mean = xtrain.mean()
std = xtrain.std()

train_loader = DataLoader(TempClassificationDataset(xtrain, length, mean, std),
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)
test_loader = DataLoader(TempClassificationDataset(xtest,length, mean, std),
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)


# --- Model

model = RNN(1,nb_villes, latent_size).to(device)
h = torch.zeros(batch_size, latent_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)


writer = SummaryWriter()
debut = time.time()
cpt = 0
for epoch in range(nb_epoch):
    
    print(f"-------------- TRAIN {epoch} ----------------")
    
    for x,y in train_loader:
        
        x = x.to(device)
        y = y.to(device)
        
        x = x.transpose(0,1).unsqueeze(2) # length*batch*1
        latent = model.forward(x, h)
        yhat = model.decode(latent[-1])
        loss = loss_fn(yhat, y)
        
        writer.add_scalar("loss/train", loss.item(), cpt)
        
        if cpt%100==0:
            print(f"{cpt} : {loss.item()}")
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        cpt += 1

    print(f"-------------- TEST {epoch} ----------------")
     
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in test_loader:
            
            x = x.to(device)
            y = y.to(device)
    
            latent = model.forward(x.transpose(0,1).unsqueeze(2), h)
            yhat = model.decode(latent[-1])
            correct += np.where(yhat.argmax(1)==y, 1, 0).sum()
            total += y.shape[0]
    
    print(f"accuracy test epoch {epoch}: {correct/total}")

fin = time.time()
print(f"time train : {(fin-debut)/60} min")
