import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def maskedCrossEntropy(output, target, padcar):
    """ compute cross entropy loss without padding """

    mask = torch.ones(output.shape)
    mask[target==padcar] = 0
    return torch.sum(output*mask)/torch.sum(mask)


class RNN(nn.Module):
    
    def __init__(self, dico_size, latent_size, emb_size):
        super(RNN, self).__init__()
        
        self.latent_size = latent_size
        
        # embedding
        self.embedding = nn.Linear(dico_size, emb_size)
        
        # encoder
        self.encoder = nn.Sequential(
                        nn.Linear(emb_size+latent_size, latent_size),
                        nn.Tanh()
                       )
        
        # decoder
        self.decoder = nn.Linear(latent_size, dico_size)
        
    def one_step(self, x, h):
        
        # x : batch_size*emb_size
        # h : batch_size*latent_size

        # output : batch_size*latent_size
        
        concat = torch.cat((x, h), dim=1)
        return self.encoder(concat)
        
    def forward(self, x, h):
        
        # x : length*batch_size
        # h : batch_size*latent_size

        # output : length*batch_size*latent_size
        
        l = []
        x = self.embedding(x)
        for t in range(x.shape[0]):
            
            h_new = self.one_step(x[t], h)
            l.append(h_new)
            h = h_new

        return torch.stack(l)
    
    def decode(self, h):

        # h : batch_size*latent_size
        # output : batch_size*dico_size

        return self.decoder(h)


class GRU(nn.Module):
    
    def __init__(self, dico_size, latent_size, emb_size):
        
        super(GRU, self).__init__()
        
        self.latent_size = latent_size
        
        self.embedding = nn.Embedding(dico_size, emb_size)
        
        # z => update gate : how much of the previous memory to keep around
        # r => reset gate : how to combine the new input with the previous memory
        self.linear_z = nn.Linear(emb_size+latent_size, latent_size)
        self.linear_r = nn.Linear(emb_size+latent_size, latent_size)
        self.act_gate = nn.Sigmoid()
        
        self.linear_h = nn.Linear(emb_size+latent_size, latent_size)
        self.act_encode = nn.Tanh()
        
        # decoder
        self.decoder = nn.Linear(latent_size, dico_size)
        
    def one_step(self, x, h):
        
        # x : batch_size*emb_size
        # h : batch_size*latent_size
        
        # output : batch_size*latent_size
        
        concat = torch.cat((x, h), dim=1)
        z = self.act_gate(self.linear_z(concat))
        r = self.act_gate(self.linear_r(concat))
        
        concat = torch.cat((x,r*h), dim=1)
        return (1-z)*h + z*self.act_encode(self.linear_h(concat))
    
    def forward(self, x, h):
        
        # x : length*batch_size
        # h : batch_size*latent_size

        # output : length*batch_size*latent_size
        
        l = []
        x = self.embedding(x)
        for t in range(x.shape[0]):
            
            h_new = self.one_step(x[t], h)
            l.append(h_new)
            h = h_new
            
        return torch.stack(l)
    
    def decode(self, h):

        # h : batch_size*latent_size
        # output : batch_size*dico_size

        return self.decoder(h)


class LSTM(nn.Module):
    
    def __init__(self, dico_size, latent_size, emb_size):
        
        super(LSTM, self).__init__()
        
        self.latent_size = latent_size
        
        self.embedding = nn.Embedding(dico_size, emb_size)
        
        self.linear_f = nn.Linear(emb_size+latent_size, latent_size)
        self.linear_i = nn.Linear(emb_size+latent_size, latent_size)
        self.linear_o = nn.Linear(emb_size+latent_size, latent_size)
        self.act_gate = nn.Sigmoid()
        
        self.linear_c = nn.Linear(emb_size+latent_size, latent_size)
        self.act_encode = nn.Tanh()
        
        # decoder
        self.decoder = nn.Linear(latent_size, dico_size)

    def one_step(self, x, h):
        
        # x : batch_size*emb_size
        # h : batch_size*latent_size
        
        # output : batch_size*latent_size

        concat = torch.cat((x, h), dim=1)
        f = self.act_gate(self.linear_f(concat))
        i = self.act_gate(self.linear_i(concat))
        self.c = f*self.c + i*self.act_encode(self.linear_c(concat))
        
        o = self.act_gate(self.linear_o(concat))
        return o*self.act_encode(self.c)
    
    def forward(self, x, h):
        
        # x : length*batch_size
        # h : batch_size*latent_size
        
        # output : length*batch_size*latent_size
        
        l = []
        x = self.embedding(x)
        self.c = torch.zeros(h.shape).to(device)
        for t in range(x.shape[0]):
            
            h_new = self.one_step(x[t], h)
            l.append(h_new)
            h = h_new
            
        return torch.stack(l)
    
    def decode(self, h):

        # h : batch_size*latent_size
        # output : batch_size*dico_size

        return self.decoder(h)