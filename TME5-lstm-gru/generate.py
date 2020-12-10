from textloader import code2string, string2code,id2lettre
import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(model, emb, decoder, eos, type_gen = 'random', start="", maxlen=200):
    
    """
    si type_gen = 'random' : on choisit le caractère suivant selon la 
                    distribution de probabilté obtenue à chaque pas t
    si type_gen = 'argmax' : on choisit le caractère le plus probable à chaque 
                    temps t
    """
    
    # if start of sequence is empty, choose a capital letter randomly
    if start=="":
        start = id2lettre[np.random.choice(range(28,54))]

    # encoding start of sequence
    x = torch.LongTensor(string2code(start)).view(-1,1).to(device)

    # forward
    h = torch.zeros(1, model.latent_size).to(device)
    h = model.forward(x, h)[-1] # 1*latent_size

    softmax = nn.Softmax(dim=1)

    for i in range(maxlen):

        yhat = decoder(h) # vecteur de taille dico_size
    
        if type_gen == 'random':
            # lettre suivante choisie selon la distribution de proba
            probas = softmax(yhat).cpu().view(-1).detach().numpy()
            ind = torch.tensor(np.random.choice(len(probas), p=probas)).view(1).to(device)
        else:
            ind = yhat.argmax().view(1).to(device) # indice du symbole le plus probable
        
        if ind==eos:
            break
        
        # on rajoute le symbole choisi à la suite de la séquence
        start += code2string(ind)
        
        h = model.one_step(emb(ind), h)
        
    return start


def generate_beam(model, emb, decoder, eos, k, start="", maxlen=200):

    # if start of sequence is empty, choose a capital letter randomly
    if start=="":
        start = id2lettre[np.random.choice(range(28,54))]
    
    # encoding start of sequence
    x = torch.LongTensor(string2code(start)).view(-1,1).to(device)
    
    # forward
    h = torch.zeros(1, model.latent_size).to(device)
    h = model.forward(x, h)[-1] # 1*latent_size
    
    # decoding
    yhat = model.decode(h) # k*dico_size
    ids = yhat.argsort(descending=True)[0,:k].view(1,k) # k ids most likely
    probas = yhat[0, ids].view(k,1)
    
    # concatenation so we can compute for the k best choices at the same time
    h = torch.cat(k*[h]) # k*latent_size
    
    for i in range(maxlen):
    
        h = model.one_step(emb(ids[-1]), h)
        yhat = decoder(h) # k*dico_size
        
        # update join proba
        new_probas = yhat+probas
        
        # get k best choices
        sorted, indices = torch.sort(new_probas,descending=True)
        ind = sorted[:,:k].flatten().argsort(descending=True)[:k]
        rows = ind//k
        best_k = indices[:,:k].flatten()[ind].view(1,k)
        
        # update
        ids = ids[:,rows]
        ids = torch.cat((ids,best_k))
        
        probas = new_probas[rows, best_k].view(k,1)
    
    res = []
    for i in range(k):
        res.append((start + code2string(ids[:,i])).split("|")[0])

    return res


def p_nucleus(decoder, k: int):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        decoder: renvoie les logits étant donné l'état du RNN
        k (int): [description]
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
            h (torch.Tensor): L'état à décoder
        """
        
        yhat = decoder(h)
        sorted, indices = torch.sort(yhat,descending=True) # 1*dico_size
        probas = sorted[0,:k]
        probas /= probas.sum()
        
        return probas, indices[0,:k]
        
    return compute


def generate_nucleus_sampling(model, emb, decoder, k, eos, start="", maxlen=200):

    # if start of sequence is empty, choose a capital letter randomly
    if start=="":
        start = id2lettre[np.random.choice(range(28,54))]
    
    # encoding start of sequence
    x = torch.LongTensor(string2code(start)).view(-1,1).to(device)
    
    # forward
    h = torch.zeros(1, model.latent_size).to(device)
    h = model.forward(x, h)[-1] # 1*latent_size
    
    nucleus = p_nucleus(decoder, k)
    
    for i in range(maxlen):
    
        probas, indices = nucleus(h)
        probas = probas.cpu().detach().numpy()
        
        # lettre suivante choisie selon le nucleus sampling
        ind = indices[np.random.choice(len(probas), p=probas)].view(1).to(device)
            
        if ind==eos:
            break
        
        # on rajoute le symbole choisi à la suite de la séquence
        start += code2string(ind)
        
        h = model.one_step(emb(ind), h)
        
    return start
