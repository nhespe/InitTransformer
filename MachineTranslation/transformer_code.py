""" WIP: Transformer By Hand

* NOTE THIS DOES NOT WORK YET, THERE IS NOT ENOUGH TIME IN THE DAY *
	
FOLLOWS THE FOLLOWING CLASS LAYOUT

[NOT DONE] class Transformer
	[NOT DONE] class Encoder:
	[NOT DONE] class Decoder:
		<CHECK> class Embedder: Given word return vector rep
		<CHECK> class PositionalEncoder: Positional Encoding of the words 
		<CHECK> class MultiHeadAttention: Calculates the attention values
		[NOT DONE] class Normalization:
		[NOT DONE] class FeedForward:

Following by a walk-through by https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
"""
from typing import List
import copy, math, time

from torch import nn, Tensor
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Embedder(nn.Module):
	""" Embeds tokens based on word indice -> vector indice 
        
        "What does the word mean"
        Input sentences are represented as 2-D matrices, of (Word, Embedding)
    """
    def __init__(self, vocab_size: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim) # lookup betwen indices and vectors

    def forward(self, x_i): # these "vectors" are trained via GD as params for the model
        return self.embed(x_i)

class PositionalEncoder(nn.Module):
    """ What is a words position in a sentence? 

        Input sentences are represented as 2-D matrices, of (Pos, i) where:
            Pos refers to the words location in the sentence
            'i' is the position within the embedding vectors

        where pos_encoding(pos, 2i) = sin(pos/10000^(2i/model_dim))
        AND where pos_encoding(pos, 2i+1) = cos(pos/10000^(2i/model_dim))
    """
    def __init__(self, model_dim: int, max_seq: int=80):
        super().__init__()
        self.model_dim = model_dim

        pos_encoding = torch.zeros(max_seq, model_dim) # pos-enconding matrix constant
        
        for pos in range(max_seq): 
            for i in range(0, model_dim, 2): # this process goes by two's -> reflecting the diff in sin, and cos for 2i, 2i+1 -> better than an "if even do sin, else cos . . . "
                pos_encoding[pos,i] = math.sin(pos/(10000**((2*i)/model_dim)))
                pos_encoding[pos,i+1] = math.cos(pos/(10000**((2*(i+1))/model_dim)))
                
        pos_encoding = pos_encoding.unsqueeze(0) # unsqueeze(zero) literally just adds it to another tensor - e.g. 1,4 -> 1,1,4

        self.register_buffer('pos_encoding', pos_encoding) # still unsure what this is about, but they instist is necessary? To my knowledge its now just a parameter of the model?
 
    
    def forward(self, word_emb):
        ## You have to boost the word embedding so that its not lost when added to the positional encoding
        word_emb = word_emb * math.sqrt(self.model_dim)

        # Return the combined word embedding with the positional embedding as a tensor
        seq_len = word_emb.size(1)
        return word_emb + Variable(self.pos_encoding[:,:seq_len], requires_grad=False) # NOTE TO SELF, REQUIRES_GRAD=FALSE MIGHT BREAK BACK PROP 

class Masking():
    """ Create using the Masker(TEXT_A.vocab.stoi["<pad>"], TEXT_A.vocab.stoi["<pad>"])

    """
    def __init__(self, input_pad: str, target_pad: str):
        self.input_pad = input_pad
        self.target_pad = target_pad

    def mask_input(seq_batch):
        """ This assumes we are getting a batch from our next(Iterator) 
        from the ProcessSequence object

        we want to create a mask with 0s where there is padding in the input sequence

        """
        input_seq = seq_batch.English.transpose(0,1)
        input_msk = (input_seq != self.input_pad).unsqueeze(1) # -> reduces [1,2,3,4] to [[1], [2], [3]. . .]
        return input_mask

    def mask_target(seq_batch: Tensor):
        """ """
        target_seq = seq_batch.French.transpose(0,1)
        target_msk = (target_seq != self.target_pad).unsqueeze(1)

        size = target_seq.size(1) # get seq_len for matrix
        nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8') # lower triange matrix of 1's
        nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0) # Turn to boolean
        target_msk = target_msk & nopeak_mask  # BITWISE AND -> NOT SURE WHY THEY DID THIS???????????

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, model_dim: int, dropout:float=0.1):
        super().__init__()
        
        self.model_dim = model_dim
        self.heads = heads
        self.k_dim = model_dim // heads
        
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.out_layer = nn.Linear(model_dim, model_dim)
    
    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask=None):
        """ 
            PROCESS: 
                <IN>
                Linear Translation
                Scaled dot Product
                Concatenation
                Linear Tranlastion
                <OUT>
        """
        # applies a linear translation to (q, k, v) (AS PER AIAYN Diagram)
        k = self.k_linear(k).view(q.size(0), -1, self.heads, self.k_dim).transpose(1,2)
        q = self.q_linear(q).view(q.size(0), -1, self.heads, self.k_dim).transpose(1,2)
        v = self.v_linear(v).view(q.size(0), -1, self.heads, self.k_dim).transpose(1,2)

        scores = attention(q, k, v, self.k_dim, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(q.size(0), -1, self.model_dim)        
        return self.out_layer(concat)

    def attention(q:Tensor, k:Tensor, v:Tensor, k_dim:int, mask:str=None, dropout=None):
        """ Key, Query, Value   
            as per paper -> Attention(q, v, k) = Softmax((QK^T) / math.sqrt(d_k))*V
                     
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k_dim)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -np.inf)
            scores = F.softmax(scores, dim=-1) # yields the key that matches the query (per vector calc - similar directions)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v) # returns the value from the found key
        return output

## NEXT TO DO
"""class Normalizer():
    def __init__(self)"""
