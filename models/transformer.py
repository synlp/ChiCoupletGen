import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def sequence_mask(sequence_length, max_len=None):   
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long() 
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len) 
    seq_range_expand = torch.autograd.Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(log_probs, target, length):  
    """
    Args:
        log_probs: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """
    length = torch.autograd.Variable(torch.LongTensor(length))
    if torch.cuda.is_available():
        length = length.cuda()

    log_probs_flat = log_probs.view(-1, log_probs.size(-1)) ## -1 means infered from other dimentions 
    target_flat = target.contiguous().view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))  
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def perplexity(log_probs, target, length):  
    """
    Args:
        log_probs: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        perplexity: An average perplexity value masked by the length.
    """
    length = torch.autograd.Variable(torch.LongTensor(length))
    if torch.cuda.is_available():
        length = length.cuda()

    log_probs_flat = log_probs.view(-1, log_probs.size(-1)) ## -1 means infered from other dimentions 
    target_flat = target.contiguous().view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))  
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    perplex = math.e ** loss.item()
    return perplex


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = embed
        self.tgt_embed = embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # print(x)
        return self.lut(x) * math.sqrt(self.d_model)

class PretrainedEmbeddings(nn.Module):
    def __init__(self, d_model, pretr_dict: torch.Tensor):
        super(PretrainedEmbeddings, self).__init__()
        assert d_model == pretr_dict.size(1)
        self.d_model = d_model
        self.special_tokens = nn.Parameter(torch.randn((5, d_model)))
        self.pretrained = pretr_dict.requires_grad_(False)
        
    def forward(self, x):
        weights = torch.cat([self.special_tokens,
                             self.pretrained], dim=0)
        return F.embedding(x, weights) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


def make_model(vocab_tensor, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(PretrainedEmbeddings(d_model, vocab_tensor), c(position)),
        Generator(d_model, vocab_tensor.size(0) + 5))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class LearnablePE(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(LearnablePE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.factor = nn.Parameter(torch.tensor(1.))   ####
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)*self.factor ####
        return self.dropout(x)

class LearnedPE(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(LearnedPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = nn.Parameter(torch.zeros(max_len, d_model))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        se = nn.Parameter(torch.zeros(2, d_model))
        self.register_buffer('se', se)
        
    def forward(self, x, length):
        max_len = x.size(1) - 1
        x[:, 1:, :] += self.pe[:max_len, :]
        for i in range(x.size(0)):
            end = 1 + length[i]
            x[i, 1:end, :] += self.se[0, :]
            x[i, end:, :] += self.se[1, :]
        return self.dropout(x)

class OnewayModel(nn.Module):
    "The model is a encoder with generator."
    def __init__(self, encoder, embed, generator):
        super(OnewayModel, self).__init__()
        self.encoder = encoder
        self.embed = embed
        self.generator = generator
        
    def forward(self, src, mask):
        "Pass the input (and mask) through each layer in turn."
        return self.encoder(self.embed(src), mask)

def make_OnewayModel(vocab_tensor, N=6, 
                 d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = LearnablePE(d_model, dropout)
    model = OnewayModel(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(PretrainedEmbeddings(d_model, vocab_tensor), c(position)),
        Generator(d_model, vocab_tensor.size(0) + 5))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def make_CoherenceModel(vocab_tensor, N=6, 
                 d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = LearnedPE(d_model, dropout)
    model = OnewayModel(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(PretrainedEmbeddings(d_model, vocab_tensor), c(position)),
        nn.Linear(d_model, 1))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class TwowayModel(nn.Module):
    "The model is a decoder with generator."
    def __init__(self, decoder, embed, generator):
        super(TwowayModel, self).__init__()
        self.decoder = decoder
        self.embed = embed
        self.generator = generator
        
    def forward(self, context, src, context_mask, src_mask):
        "Pass the input (and mask) through each layer in turn."
        return self.decoder(self.embed(src), 
                            self.embed(context), 
                            context_mask, 
                            src_mask)

def make_TwowayModel(vocab_tensor, N=6, 
                 d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = LearnablePE(d_model, dropout)
    model = TwowayModel(
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(PretrainedEmbeddings(d_model, vocab_tensor), c(position)),
        Generator(d_model, vocab_tensor.size(0) + 5))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class Memory(nn.Module):
    "Generic N layer memory net with masking."
    def __init__(self, layer, N):
        super(Memory, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask)
        return self.norm(x)

class MemoryLayer(nn.Module):
    "Memory net is made of mem-attn, and feed forward (defined below)"
    def __init__(self, size, src_attn, feed_forward, dropout):
        super(MemoryLayer, self).__init__()
        self.size = size
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
 
    def forward(self, x, memory, src_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[1](x, self.feed_forward)
        

class MemoryModel(nn.Module):
    "The model is a encoder with generator and memory."
    def __init__(self, encoder, memory, embed, generator):
        super(MemoryModel, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.embed = embed
        self.generator = generator
        
    def forward(self, context, src, context_mask, src_mask):
        "Pass the input (and mask) through each layer in turn."
        return self.memory(self.encoder(self.embed(src), src_mask),
                           self.embed(context),
                           context_mask)

def make_MemoryModel(vocab_tensor, N=6, 
                     d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = LearnablePE(d_model, dropout)
    model = MemoryModel(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Memory(MemoryLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(PretrainedEmbeddings(d_model, vocab_tensor), c(position)),
        Generator(d_model, vocab_tensor.size(0) + 5))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class ConditionalDecoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(ConditionalDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        
    def forward(self, x, context, src_mask, tgt_mask, memory):
        for layer in self.layers:
            x = layer(x, context, src_mask, tgt_mask, memory)
        return self.norm(x)

class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(d_model, d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # import pdb; pdb.set_trace()
        delta_gamma = self.mlp_gamma(memory)
        delta_beta = self.mlp_beta(memory)
        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat

class ConditionalDecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots):
        super(ConditionalDecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        return self.sublayer[2](x, self.feed_forward, memory)


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)))


class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

        self.Wk = nn.Linear(self.d_model + self.d_model, self.d_model)
        self.Wv = nn.Linear(self.d_model + self.d_model, self.d_model)

    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1).expand(-1, memory.size(1), -1)], 2)
        v = torch.cat([memory, input.unsqueeze(1).expand(-1, memory.size(1), -1)], 2)
        # import pdb; pdb.set_trace()
        k = self.Wk(k)
        v = self.Wv(v)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        # import pdb; pdb.set_trace()
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class RelationalMemoryModel(nn.Module):

    def __init__(self, encoder, decoder, embed, generator, rm) -> None:
        super(RelationalMemoryModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = embed
        self.tgt_embed = embed
        self.generator = generator
        self.rm = rm
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = self.rm(self.tgt_embed(tgt), memory)
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)

def make_ReMModel(vocab_tensor, N=6, 
                  d_model=512, d_ff=2048, h=8, dropout=0.1,
                  num_slot=4):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = LearnablePE(d_model, dropout)
    model = RelationalMemoryModel(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        ConditionalDecoder(ConditionalDecoderLayer(d_model, c(attn), c(attn), c(ff), dropout, num_slot), N),
        nn.Sequential(PretrainedEmbeddings(d_model, vocab_tensor), c(position)),
        Generator(d_model, vocab_tensor.size(0) + 5),
        RelationalMemory(num_slots=num_slot, d_model=d_model, num_heads=h))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class BertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator):
        super(BertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = bert.embeddings
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask, return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)

def make_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = BertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model, vocab)
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class SegBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator):
        super(SegBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = bert.embeddings
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask, return_hidden=False):
        "Take in and process masked src and target sequences."
        hidden = self.encode(src, src_mask)
        if return_hidden:
            return self.decode(hidden, src_mask, tgt, tgt_mask), hidden
        return self.decode(hidden, src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask, return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)

def make_seg_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = SegBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model, vocab)
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class POSBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator, pos_embed):
        super(POSBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = bert.embeddings
        self.generator = generator
        self.pos_embed = pos_embed
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask, return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)

def make_pos_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1, d_pos=32):
    c = copy.deepcopy
    vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = POSBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model + d_pos, vocab),
        nn.Embedding(33, d_pos)
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.pos_embed.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class SegPOSBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator, pos_embed):
        super(SegPOSBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = bert.embeddings
        self.generator = generator
        self.pos_embed = pos_embed
        
    def forward(self, src, tgt, src_mask, tgt_mask, return_hidden=False):
        "Take in and process masked src and target sequences."
        hidden = self.encode(src, src_mask)
        if return_hidden:
            return self.decode(hidden, src_mask, tgt, tgt_mask), hidden
        return self.decode(hidden, src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask, return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)

def make_seg_pos_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1, d_pos=32):
    c = copy.deepcopy
    vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = SegPOSBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model + d_pos, vocab),
        nn.Embedding(33, d_pos)
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.pos_embed.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class TestSegPOSBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator, pos_embed, encoder_add, decoder_add):
        super(TestSegPOSBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = bert.embeddings
        self.generator = generator
        self.pos_embed = pos_embed
        self.encoder_add = encoder_add
        self.decoder_add = decoder_add
        
    def forward(self, src, tgt, src_mask, tgt_mask, pos, return_hidden=False):
        "Take in and process masked src and target sequences."
        hidden = self.encode(src, src_mask, pos)
        if return_hidden:
            return self.decode(hidden, src_mask, tgt, tgt_mask, pos), hidden
        return self.decode(hidden, src_mask, tgt, tgt_mask, pos)
    
    def encode(self, src, src_mask, pos):
        if self.encoder_add:
            batch_size = src.size(0)
            pos = torch.cat([torch.tensor([[0]]*batch_size).cuda(), pos], dim=1)
            src_embed = self.encoder.embeddings(src)
            pos_embed = self.pos_embed(pos)
            embed = src_embed + pos_embed
            return self.encoder(attention_mask=src_mask, inputs_embeds=embed, return_dict=True)['last_hidden_state']
        return self.encoder(src, src_mask, return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask, pos):
        if self.decoder_add:
            length = tgt.size(1)
            tgt_embed = self.tgt_embed(tgt)
            pos_embed = self.pos_embed(pos[:, :length])
            embed = tgt_embed + pos_embed
            return self.decoder(embed, memory, src_mask.unsqueeze(-2), tgt_mask)
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)

def make_seg_pos_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1, d_pos=32):
    c = copy.deepcopy
    vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = SegPOSBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model + d_pos, vocab),
        nn.Embedding(33, d_pos)
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.pos_embed.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class ExpandPOSBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator, src_embed, tgt_embed, pos_embed):
        super(ExpandPOSBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.src_embed = src_embed
        self.generator = generator
        self.pos_embed = pos_embed
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(attention_mask=src_mask, 
                            inputs_embeds=self.src_embed(src), 
                            return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)


class ExpandBertEmbeddings(nn.Module):
    def __init__(self, d_model, add_vocab, pretr_dict: torch.Tensor):
        super(ExpandBertEmbeddings, self).__init__()
        assert d_model == pretr_dict.size(1)
        self.d_model = d_model
        self.add_tokens = nn.Parameter(torch.randn((add_vocab, d_model)))
        self.pretrained = nn.Parameter(pretr_dict)
        self.initialize()
        
    def forward(self, x):
        weights = torch.cat([self.pretrained,
                             self.add_tokens], dim=0)
        return F.embedding(x, weights) * math.sqrt(self.d_model)

    def initialize(self):
        nn.init.xavier_uniform_(self.add_tokens)

def make_expand_pos_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1, d_pos=32, add_vocab=0):
    c = copy.deepcopy
    bert_vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    vocab = bert_vocab + add_vocab
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    embed = nn.Sequential(ExpandBertEmbeddings(d_model, add_vocab, bert.embeddings.word_embeddings.weight),
                      PositionalEncoding(d_model, dropout))
    model = ExpandPOSBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model + d_pos, vocab),
        embed,
        embed,
        nn.Embedding(33, d_pos)
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.pos_embed.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class ExpandBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator, src_embed, tgt_embed):
        super(ExpandBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.src_embed = src_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(attention_mask=src_mask, 
                            inputs_embeds=self.src_embed(src), 
                            return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)


def make_expand_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1, d_pos=32, add_vocab=0):
    c = copy.deepcopy
    bert_vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    vocab = bert_vocab + add_vocab
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    embed = nn.Sequential(ExpandBertEmbeddings(d_model, add_vocab, bert.embeddings.word_embeddings.weight),
                      PositionalEncoding(d_model, dropout))
    model = ExpandBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model, vocab),
        embed,
        embed
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class DependencyMemory(nn.Module):

    def __init__(self, text_embed, type_embed):
        super(DependencyMemory, self).__init__()
        self.text_embed = text_embed
        self.type_embed = type_embed
        self.temper = text_embed.embedding_dim ** 0.5

    def forward(self, queries, dep_text, dep_type, memory_mask, output_mask):
        text_embed = self.text_embed(dep_text)
        type_embed = self.type_embed(dep_type)
        embed = text_embed + type_embed
        scores = torch.matmul(embed, queries.unsqueeze(-1)) / self.temper
        scores = scores.squeeze(-1)
        scores = scores.masked_fill_(memory_mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores.unsqueeze(-2), embed).squeeze()
        output = output * output_mask.unsqueeze(-1)
        return output + queries

class MemoryBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator, memory):
        super(MemoryBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = bert.embeddings
        self.generator = generator
        self.memory = memory
        
    def forward(self, src, tgt, src_mask, tgt_mask, dep_text, dep_type, memory_mask, output_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.memory(self.encode(src, src_mask), 
                                       dep_text, dep_type, memory_mask, 
                                       output_mask), 
                           src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(src, attention_mask=src_mask, 
                            return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)


def make_dep_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    memory = DependencyMemory(nn.Embedding(vocab, d_model), nn.Embedding(47, d_model))
    model = MemoryBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model, vocab),
        memory
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.memory.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class DecoderMemoryBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator, memory):
        super(DecoderMemoryBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = bert.embeddings
        self.generator = generator
        self.memory = memory
        
    def forward(self, src, tgt, src_mask, tgt_mask, dep_text, dep_type, memory_mask, output_mask):
        "Take in and process masked src and target sequences."
        hidden = self.encode(src, src_mask)
        memory = self.memory(hidden, dep_text, dep_type, memory_mask, output_mask)
        return self.decode(hidden, src_mask, tgt, tgt_mask), memory
    
    def encode(self, src, src_mask):
        return self.encoder(src, attention_mask=src_mask, 
                            return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)


def make_decoder_dep_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    memory = DependencyMemory(nn.Embedding(vocab, d_model), nn.Embedding(47, d_model))
    model = DecoderMemoryBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model*2, vocab),
        memory
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.memory.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class TwoMemoryBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator, pos_memory, dep_memory):
        super(TwoMemoryBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = bert.embeddings
        self.generator = generator
        self.pos_memory = pos_memory
        self.dep_memory = dep_memory
        d_model = self.encoder.embeddings.word_embeddings.embedding_dim
        self.W = nn.Linear(2*d_model, d_model)
        
    def forward(self, src, tgt, src_mask, tgt_mask, pos_text, pos_type, 
                pos_memory_mask, pos_output_mask, dep_text, dep_type,
                dep_memory_mask, dep_output_mask, return_hidden=False):
        "Take in and process masked src and target sequences."
        hidden = self.encode(src, src_mask)
        pos_output = self.pos_memory(hidden, pos_text, pos_type, 
                                     pos_memory_mask, pos_output_mask)
        dep_output = self.dep_memory(hidden, dep_text, dep_type, 
                                     dep_memory_mask, dep_output_mask)
        output = torch.cat([pos_output, dep_output], dim=-1)
        output = self.W(output)
        if return_hidden:
            return self.decode(output, src_mask, tgt, tgt_mask), hidden
        return self.decode(output, src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(src, attention_mask=src_mask, 
                            return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)


def make_seg_pos_dep_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pos_memory = DependencyMemory(nn.Embedding(vocab, d_model), nn.Embedding(33, d_model))
    dep_memory = DependencyMemory(nn.Embedding(vocab, d_model), nn.Embedding(47, d_model))
    model = TwoMemoryBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model, vocab),
        pos_memory,
        dep_memory
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.dep_memory.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.pos_memory.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.W.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class DecoderTwoMemoryBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator, pos_memory, dep_memory):
        super(DecoderTwoMemoryBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = bert.embeddings
        self.generator = generator
        self.pos_memory = pos_memory
        self.dep_memory = dep_memory
        
    def forward(self, src, tgt, src_mask, tgt_mask, pos_text, pos_type, 
                pos_memory_mask, pos_output_mask, dep_text, dep_type,
                dep_memory_mask, dep_output_mask, return_hidden=False):
        "Take in and process masked src and target sequences."
        hidden = self.encode(src, src_mask)
        pos_output = self.pos_memory(hidden, pos_text, pos_type, 
                                     pos_memory_mask, pos_output_mask)
        dep_output = self.dep_memory(hidden, dep_text, dep_type, 
                                     dep_memory_mask, dep_output_mask)
        memory = torch.cat([pos_output, dep_output], dim=-1)
        output = self.decode(hidden, src_mask, tgt, tgt_mask)
        if return_hidden:
            return output, memory, hidden
        return output, memory
    
    def encode(self, src, src_mask):
        return self.encoder(src, attention_mask=src_mask, 
                            return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)


def make_decoder_seg_pos_dep_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pos_memory = DependencyMemory(nn.Embedding(vocab, d_model), nn.Embedding(33, d_model))
    dep_memory = DependencyMemory(nn.Embedding(vocab, d_model), nn.Embedding(47, d_model))
    model = DecoderTwoMemoryBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model*3, vocab),
        pos_memory,
        dep_memory
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.dep_memory.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.pos_memory.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class TestPOSBertEncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bert, decoder, generator, pos_embed, encoder_add, decoder_add):
        super(TestPOSBertEncoderDecoder, self).__init__()
        self.encoder = bert
        self.decoder = decoder
        self.tgt_embed = bert.embeddings
        self.generator = generator
        self.pos_embed = pos_embed
        self.encoder_add = encoder_add
        self.decoder_add = decoder_add
        
    def forward(self, src, tgt, src_mask, tgt_mask, pos):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask, pos), src_mask,
                            tgt, tgt_mask, pos)
    
    def encode(self, src, src_mask, pos):
        if self.encoder_add:
            batch_size = src.size(0)
            pos = torch.cat([torch.tensor([[0]]*batch_size).cuda(), pos], dim=1)
            src_embed = self.encoder.embeddings(src)
            pos_embed = self.pos_embed(pos)
            embed = src_embed + pos_embed
            return self.encoder(attention_mask=src_mask, inputs_embeds=embed, return_dict=True)['last_hidden_state']
        return self.encoder(src, src_mask, return_dict=True)['last_hidden_state']
    
    def decode(self, memory, src_mask, tgt, tgt_mask, pos):
        if self.decoder_add:
            length = tgt.size(1)
            tgt_embed = self.tgt_embed(tgt)
            pos_embed = self.pos_embed(pos[:, :length])
            embed = tgt_embed + pos_embed
            return self.decoder(embed, memory, src_mask.unsqueeze(-2), tgt_mask)
        return self.decoder(self.tgt_embed(tgt), memory, src_mask.unsqueeze(-2), tgt_mask)

def make_test_pos_bertmodel(bert, N=6, d_ff=2048, h=8, dropout=0.1, encoder_add=False, decoder_add=False):
    c = copy.deepcopy
    vocab = bert.embeddings.word_embeddings.num_embeddings
    d_model = bert.embeddings.word_embeddings.embedding_dim
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = TestPOSBertEncoderDecoder(
        bert,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model, vocab),
        nn.Embedding(33, d_model),
        encoder_add=encoder_add,
        decoder_add=decoder_add
    )
    for p in model.decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in model.pos_embed.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

