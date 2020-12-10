from utils import read_temps, RNN, TempPredictionDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# --- Load data
 
temp_train = read_temps(r"data/tempAMAL_train.csv")
temp_test = read_temps(r"data/tempAMAL_test.csv")


# =============================================================================
# ----------------------------- Modèle univarié -------------------------------
# =============================================================================

# --- Parameters

nb_villes = 5
length = 20
batch_size = 64
latent_size = 20

lr = 0.001
nb_epoch = 10


# --- Dataloader and Model

xtrain = temp_train[:,:nb_villes]
xtest = temp_test[:,:nb_villes]
mean = xtrain.mean()
std = xtrain.std()

train_loaders = []
test_loaders = []
models = [] # 1 model par ville
optims = []
for i in range(nb_villes):
    train_loaders.append(DataLoader(TempPredictionDataset(xtrain[:,i], length, mean, std),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    drop_last=True))
    test_loaders.append(DataLoader(TempPredictionDataset(xtest[:,i], length, mean, std),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    drop_last=True))
    models.append(RNN(1, 1, latent_size))
    optims.append(torch.optim.Adam(models[i].parameters(), lr=lr))

h = torch.zeros(nb_villes,batch_size,latent_size)
loss_fn = nn.MSELoss()


# --- Training

writer = SummaryWriter()
for i in range(nb_villes):
    cpt = 0
    
    for epoch in range(nb_epoch):
        
        print(f"-------------- TRAIN {epoch} Ville {i} ----------------")
        
        for x,y in train_loaders[i]:
                
            latent = models[i].forward(x.transpose(0,1).unsqueeze(2), h[i])
            
            loss = 0
            for t in range(length):
                yhat = models[i].decode(latent[t])
                loss += loss_fn(yhat, y[:,t].view(-1,1))
                
            writer.add_scalar(f"loss/train_ville{i}", loss.item(), cpt)
            
            if cpt%100==0:
                print(f"{cpt} : {loss.item()}")
            
            optims[i].zero_grad()
            loss.backward()
            optims[i].step()
            
            cpt += 1
    
        print(f"-------------- TEST {epoch} Ville {i} ----------------")
        
        with torch.no_grad():
            
            cpt_test = 0
            
            for x,y in test_loaders[i]:
        
                latent = models[i].forward(x.transpose(0,1).unsqueeze(2), h[i])
                
                loss = 0
                for t in range(length):
                    yhat = models[i].decode(latent[t])
                    loss += loss_fn(yhat, y[:,t].view(-1,1))
                
                if cpt_test%100==0:
                    print(f"{cpt_test} : {loss.item()}")
                
                cpt_test += 1
            
            print(f"mean loss test epoch {epoch} ville {i} : {loss/cpt_test}")


# =============================================================================
# ----------------------------- Modèle multivarié -----------------------------
# =============================================================================

# --- Parameters
            
nb_villes = 10
length = 20
batch_size = 64
latent_size = 20

lr = 0.01
nb_epoch = 10

# --- Dataloader

xtrain = temp_train[:,:nb_villes]
xtest = temp_test[:,:nb_villes]
mean = xtrain.mean()
std = xtrain.std()

train_loader = DataLoader(TempPredictionDataset(xtrain, length, mean, std),
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)
test_loader = DataLoader(TempPredictionDataset(xtest,length, mean, std),
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)


# --- Model

model = RNN(nb_villes, nb_villes, latent_size)
h = torch.zeros(batch_size, latent_size)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)


# --- Training

writer = SummaryWriter()
cpt = 0
for epoch in range(nb_epoch):
    
    print(f"-------------- TRAIN {epoch} ----------------")
    
    for x,y in train_loader:
            
        latent = model.forward(x.transpose(0,1), h)
        
        loss = 0
        for t in range(length):
            yhat = model.decode(latent[t])
            loss += loss_fn(yhat, y[:,t])
            
        writer.add_scalar(f"loss/train", loss.item(), cpt)
        
        if cpt%100==0:
            print(f"{cpt} : {loss.item()}")
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        cpt += 1

    print(f"-------------- TEST {epoch} ----------------")
    
    with torch.no_grad():
        
        cpt_test = 0
        
        for x,y in test_loader:
    
            latent = model.forward(x.transpose(0,1), h)
            
            loss = 0
            for t in range(length):
                yhat = model.decode(latent[t])
                loss += loss_fn(yhat, y[:,t])
            
            if cpt_test%100==0:
                print(f"{cpt_test} : {loss.item()}")
            
            cpt_test += 1
    
    print(f"epoch {epoch} mean loss test : {loss/(cpt_test*batch_size)}")

