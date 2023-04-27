import torch.nn as nn
import torch.nn.functional as F

from .layer import LayerNorm, SublayerConnection, clones


'''
解码器层，堆叠解码器层即可获得解码器
'''
class DocoderLayer(nn.Module):
    
    '''
    size:           d_model
    self_attn:      多头自注意力，目标语言
    src_attn:       多头自注意力，源语言
    feed_forward:   前馈网络层
    '''
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DocoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 带掩码的多头自注意力层            残差连接与归一化
        self.sublayer_self_attn = SublayerConnection(size, dropout)
        # 与解码器输出交互的多头自注意力层   残差连接与归一化
        self.sublayer_src_attn  = SublayerConnection(size, dropout)
        # 前馈网络                         残差连接与归一化
        self.sublayer_fw = SublayerConnection(size, dropout)


    '''
    x:          目标语言的词向量
    memory:     源语言经过解码器之后输出的词向量
    src_mask:   源语言的掩码
    tat_mask:   目标语言的掩码
    '''
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # 带掩码的多头自注意力层
        self_attn = lambda v: self.self_attn(v, v, v, tgt_mask)
        out = self.sublayer_self_attn(x, self_attn)
        # 与解码器输出交互的多头自注意力层
        src_attn = lambda v: self.src_attn(v, m, m, src_mask)
        out = self.sublayer_src_attn(x, src_attn)
        # 前馈网络层
        out = self.sublayer_fw(out)
        return out
        



'''
解码器，堆叠 N 个解码器层形成解码器
'''
class Decoder(nn.Module):
    
    '''
    layer:  解码器层  EncoderLayer
    N:      解码器层的个数
    '''
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        out = self.norm(x)
        return out



'''
解码器的输出层
由一个线性层与一个softmax层组成
'''
class Generator(nn.Module):
    
    '''
    d_model:    词向量的维度数
    vocab:      词表的大小
    '''
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # 通过前一个词预测当前词
        self.proj = nn.Linear(d_model, vocab)

    '''
    x:  [B, num, d_model]
    '''
    def forward(self, x):
        out = self.proj(x)
        out = F.log_softmax(out, dim=-1)
        return out