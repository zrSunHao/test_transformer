import torch.nn as nn
import copy

from .layer import LayerNorm, SublayerConnection

'''
编码器层，堆叠编码器层即可获得编码器
'''
class EncoderLayer(nn.Module):
    
    '''
    size:           d_model
    self_attn:      多头自注意力
    feed_forward:   前馈层
    '''
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 多头自注意力的残差连接与归一化
        self.sublayer_attn = SublayerConnection(size, dropout)
        # 前馈网络的残差连接与归一化
        self.sublayer_fw = SublayerConnection(size, dropout)

    def forward(self, x, mask):
        # 使用lambda表达式构建一个方法，输入参数v
        self_attn = lambda v: self.self_attn(v, v, v, mask)
        out = self.sublayer_attn(x, self_attn)
        out = self.sublayer_fw(x, self.feed_forward)
        return out


'''
工具函数
基于一个编码器层快速获得 N 个相同的编码器层
'''
def clones(module, N):
    layers = [copy.deepcopy(module) for _ in range(N)]
    return nn.ModuleList(layers)


'''
编码器，由多个编码器层堆叠而成
'''
class Encoder(nn.Module):

    '''
    layer:  编码器层  EncoderLayer
    N:      编码器层的个数
    '''
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(features=layer.size)  # EncoderLayer.size

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        out = self.norm(x)
        return out
