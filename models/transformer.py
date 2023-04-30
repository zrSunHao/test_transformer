import torch.nn as nn
import torch.nn.functional as F
import copy

from .attention import MultiHeadedAttention
from .layer import PositionwiseFeedForward
from .positional_encoding import Embeddings, PositionalEncoding
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder, Generator

'''
Transformer 模型
机器翻译 源语言 --> 目标语言
'''
class Transformer(nn.Module):

    '''
    encoder:    编码器
    decoder:    解码器
    src_embed:  源语言的词嵌入
    tgt_embed:  目标语言的词嵌入
    generator:  解码器的输出层
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    '''
    src:    源语言词向量
    tgt:    目标语言词向量
    '''
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        out = self.decode(memory, src_mask, tgt, tgt_mask)
        return out

    '''
    编码器编码
    src:    源语言的词向量  [B, token_num]
    mask:   源语言的掩码    [B, 1, token_num]
    '''
    def encode(self, src, mask):
        # [B, token_num] --> [B, 1, token_num]
        x = self.src_embed(src)
        memory = self.encoder(x, mask)
        return memory

    '''
    解码器解码
    memory:     编码器的输出        [B, token_num, d_model]
    src_mask:   源语言的编码        [B, 1, token_num]
    tgt:        目标语言的词向量    [B, token_num]
    tgt_mask:   目标语言的掩码      [B, token_num, token_num]
    '''
    def decode(self, memory, src_mask, tgt, tgt_mask):
        # [B, token_num] --> [B, token_num, d_model]
        x = self.tgt_embed(tgt)
        out = self.decoder(x, memory, src_mask, tgt_mask)
        return out



'''
构造网络
    src_vocab:  源语言词表的大小
    tgt_vocab:  目标语言词表的大小
    N:          解码器层和编码器层的个数
    d_model:    词向量的大小
    d_ff:       前馈网络层中间维度的大小
    h:          多头注意力中头的个数
'''
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, 
               dropout=0.1, device='cpu'):
    c = copy.deepcopy

    # 实例化多头注意力
    attn = MultiHeadedAttention(h, d_model)
    # 实例化前馈网络层
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化位置编码
    position = PositionalEncoding(d_model, dropout)

    # 实例化编码层
    encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
    # 实例化编码器
    encoder = Encoder(encoder_layer, N)

    # 实例化解码层
    decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
    # 实例化解码器
    decoder = Decoder(decoder_layer, N)

    # 实例化编码器词嵌入
    src_embed = nn.Sequential(
        Embeddings(d_model, src_vocab),
        c(position)
    )
    # 实例化解码器词嵌入
    tgt_embed = nn.Sequential(
        Embeddings(d_model, tgt_vocab),
        c(position)
    )

    # 实例化解码器的输出层
    generator = Generator(d_model, tgt_vocab)

    # 实例化模型
    model = Transformer(encoder, decoder, src_embed, tgt_embed, generator)
    model = model.to(device)
    # 初始化参数，重要，xavier_uniform 是一种初始化方法
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model