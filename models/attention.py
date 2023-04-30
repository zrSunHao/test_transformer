import torch.nn as nn
import torch as t
import torch.nn.functional as F
import math

'''
实现注意力操作
输入:   query, key, value    [B, num, d_k]
输出:   out 相当于每个token 的新表示， 即value
        p_attn 每个 token 的 query 其他 token 的 key 相乘的得分
'''
def attention(query, key, value, mask=None, dropout=None):
    
    d_k = key.size(-1)          # 每个key 的长度
    # key 的转置，这里通过交换 key 的维度实现
    t_k = key.transpose(-2, -1) # [B, num, d_k] --> [B, d_k, num]
    # query * T_key / sqrt(d_k) 计算相似度得分 [B, num, num]
    scores = t.matmul(query, t_k) / math.sqrt(d_k)
    
    if mask is not None:
        # masked_fill 第一个参数： 判断条件，第二个参数：判断条件为true的替换为该值
        scores = scores.masked_fill(mask==0, -1e9)
    
    # 归一化指数层，将分数转为概率   [B, num, num]
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # [B, num, num] * [B, num, 64] --> [B, num, 64]
    # 将句子中与每个token求得的概率与对应的value相乘，再相加，得到每个token的新表示
    out = t.matmul(p_attn, value)
    return out, p_attn



# 多头自注意力
class MultiHeadedAttention(nn.Module):

    '''
    h:  head的个数
        即大的词向量分裂为头中小的词向量的个数
    d_model:    词向量的长度
    '''
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0     # 词向量的长度可以整除头的个数

        self.d_k = d_model // h     # 小token的维度
        self.h = h                  # 头的个数
        # 512 --> 512
        self.W_Q = nn.Linear(d_model, h * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, h * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, h * self.d_k, bias=False)
        self.fc  = nn.Linear(d_model, h * self.d_k, bias=False)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        '''
        在解码器中，所有的头应用同样的 mask
        '''
        if mask is not None:
            mask = mask.unsqueeze(1)    # 扩展一个维度
        nbatches = query.size(0)        # 句子数

        '''
        1、 在一个 batch 上做线性映射，来完成多头以及 q,k,v 的生成
            d_model -->  d_k * h
        '''
        q = self.W_Q(query)             # [B, num, 512] * [B, num, h*d_k]
        # [B, num, d_k*h] --> [B, num, h, d_k] --> [B, h, num, d_k]
        q = q.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        k = self.W_K(key)
        k = k.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        v = self.W_V(value)
        v = v.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        '''
        2、 同时完成多头注意力的计算
        '''
        x, self.attn = attention(q, k, v, mask, self.dropout)

        '''
        3、拼接所有头的输出，然后做线性映射，将维度变换为 d_model
        '''
        # [B, h, num, d_k] --> [B, num, h, d_k]
        x = x.transpose(1, 2).contiguous()
        # [B, num, h, d_k] --> [B, num, h*d_k] --> [B, num, d_model]
        x = x.view(nbatches, -1, self.h * self.d_k)

        out = self.fc(x)
        return out
        


