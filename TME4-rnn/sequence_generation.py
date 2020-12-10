from utils import SpeechDataset, string2code, code2string, id2lettre
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# ----------------------- RNN avec embedding (linear) -------------------------
# =============================================================================

class RNN(nn.Module):
    
    def __init__(self, in_size, out_size, latent_size, embedding_size):
        super(RNN, self).__init__()
        
        self.embedding = nn.Linear(in_size, embedding_size)
        
        self.act1 = nn.Tanh()
        self.linear_x = nn.Linear(embedding_size,latent_size,bias=False)
        self.linear_h = nn.Linear(latent_size,latent_size)
        
        self.decoder = nn.Linear(latent_size,out_size)
        
    def one_step(self, x_t, h):
        
        # x_t : batch x dim
        # h : batch x latent
        
        # output : batch x latent
        
        return self.act1(self.linear_x(self.embedding(x_t))+self.linear_h(h))
        
    def forward(self, x, h):
        
        # x : length x batch x dim
        # h : batch x latent
        
        # output : length x batch x latent
        
        l = []
        for t in range(x.shape[0]):
            h_new = self.one_step(x[t],h) # batch x latent
            l.append(h_new)
            h = h_new
        return torch.stack(l)
    
    def decode(self, h):
        return self.decoder(h) # batch x output_size

 
# =============================================================================
# ---------------------------- Data Preparation -------------------------------
# =============================================================================
        
# --- Load data
        
with open ("data/trump_full_speech.txt","r") as f:
    text = f.read()
    
# --- One-hot encoding
    
dico_size = len(id2lettre)
labels = string2code(text)
text_onehot = torch.zeros(len(labels), dico_size)
text_onehot[range(len(labels)),labels] = 1


# =============================================================================
# -------------------------------- Training -----------------------------------
# =============================================================================

# --- Parameters

embedding_size = 80
length = 20
batch_size = 100
latent_size = 20

nb_epoch = 1
lr = 0.001


dataloader = DataLoader(SpeechDataset(text_onehot, labels, length), batch_size=batch_size, shuffle=True, drop_last=True)

model = RNN(dico_size, dico_size, latent_size, embedding_size)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)


writer = SummaryWriter()
debut = time.time()
cpt = 0
for epoch in range(nb_epoch):
    
    print(f"-------------- EPOCH {epoch} ----------------")
    
    for x,y in dataloader:
        
        x = x.transpose(0,1).view(length, batch_size, dico_size)
        h = torch.zeros(batch_size, latent_size)
        
        latent = model.forward(x,h)
        
        loss = 0
        for i in range(length):
            yhat = model.decode(latent[i])
            loss += loss_fn(yhat,y[:,i])
            
        if cpt%100==0:
            print(f"{cpt} : {loss.item()}")
           
        writer.add_scalar("loss/train", loss.item(), cpt)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        cpt += 1
    
fin = time.time()
print(f"time : {fin-debut}")


# =============================================================================
# --------------------------- Sequence generation -----------------------------
# =============================================================================


# --- type generation

# - si 'random' : on choisit le prochain caractère en fonction de la distribution
# de probabilité
# - si 'argamx' : on prend le caractère le plus probable
type_gen = 'random'


# start of sequence
seq = "We have people that "
l = string2code(seq)

# one-hot encoding
x = torch.zeros(len(l), dico_size)
x[range(len(l)),l] = 1

h = torch.zeros(1, latent_size)
h = model.forward(x,h)[-1]

softmax = nn.Softmax()

for i in range(100):
    
    yhat = model.decode(h) # vecteur de taille dico_size
    
    if type_gen=='random':
        # lettre suivante choisie selon la distribution de proba
        probas = softmax(yhat).view(-1).detach().numpy()
        ind = torch.tensor(np.random.choice(dico_size, p=probas))
    
    if type_gen=='argmax':
        ind = yhat.argmax() # indice du symbole le plus probable
        
    # on rajoute le symbole choisi à la suite de la séquence
    seq += code2string(ind.view(1))
    
    # encodage one-hot de la prédiction
    yhat_onehot = torch.zeros(1, dico_size)
    yhat_onehot[0, ind] = 1
    
    h = model.one_step(yhat_onehot, h)
    
print(seq)
