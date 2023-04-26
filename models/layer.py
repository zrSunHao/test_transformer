import torch.nn as nn
import torch as t
import torch.nn.functional as F
import math

'''
层归一化，见公式
'''
class LayerNorm(nn.Module):

    '''
    features: d_model
    '''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(t.ones(features))
        self.b_2 = nn.Parameter(t.zeros(features))
        self.eps = eps

    '''
    x: [B, num, d_model]
    '''
    def forward(self, x):
        # 求均值   mean: [B, num, 1]
        mean = x.mean(-1, keepdim=True)
        # 求标准差 mean: [B, num, 1]
        std = x.std(-1, keepdim=True)

        out = self.a_2 * (x - mean) / (std - self.eps) + self.b_2
        return out





'''
残差连接
'''
class SublayerConnection(nn.Module):
    
    '''
    size: d_model
    '''
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    '''
    x:          原数据  [B, num, d_model]
    sublayer:   原数据需经过的网络层
    '''
    def forward(self, x, sublayer):
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)
        
        out = x + res
        return out

        



'''
前向网络层
'''
class PositionwiseFeedForward(nn.Module):
    
    '''
    d_model:    模型输入词向量的维度数
    d_ff：      前向网络层的中间维度数
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.w_2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)

    '''
    x:  [B, num, d_model]
    '''
    def forward(self, x):
        out = self.w_1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.w_2(out)

        return out


