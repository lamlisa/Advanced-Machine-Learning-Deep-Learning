from torch.utils.data import Dataset
import unicodedata
import string
from typing import List
import torch

PAD_IX = 0
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '' ##NULL CHARACTER
id2lettre[EOS_IX] = '|'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def normalize(s):
    """ enlève les accents et les majuscules """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return [lettre2id[c] for c in normalize(s)]

def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)
 

class TextDataset(Dataset):
    
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        
        # split with '.' separator, encode each sentence and add EOS_IX
        # transform to LongTensor
        # => List[ LongTensor[int] ]
        self.sequences = [torch.LongTensor(string2code(sentence.strip())+[EOS_IX]) for sentence in text.split('.')]
        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        # output : sequence, length of the sequence
        return self.sequences[i], len(self.sequences[i])


def collate_fn(samples: List[List[int]]):
    """ return padded batch """ 
    
    sentences, lengths = zip(*samples)
    
    batch = torch.zeros(max(lengths), len(samples)).long()
    for i in range(len(samples)):
        batch[:lengths[i], i] = sentences[i]
        
    return batch