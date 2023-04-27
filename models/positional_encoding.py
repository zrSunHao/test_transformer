import torch.nn as nn
import torch as t
import math


'''
词嵌入
'''
class Embeddings(nn.Module):
   
    '''
    d_model:    词向量的大小
    vocab:      词表的大小
    '''
    def __init__(self, d_model, vocab):
       super(Embeddings, self).__init__()
       self.lut = nn.Embedding(num_embeddings=vocab,
                               embedding_dim=d_model)
       self.d_model = d_model

    def forward(self, x):
        a = math.sqrt(self.d_model)
        out = self.lut(x) * a
        return out



'''
词向量添加位置编码
'''
class PositionalEncoding(nn.Module):

    '''
    d_model:    模型的输入维度（词向量的维度）
    droupout:   droupout 的概率
                作为训练神经网络的一种trick选择,
                在每个训练批次中,以某种概率忽略一定数量的神经元.
                可以明显地减少过拟合现象.
    max_len:    句子的最大长度
    '''
    def __init__(self, d_model, droupout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=droupout)

        # 计算位置编码值
        # 生成 [max_len, d_model] 大小的矩阵
        pe = t.zeros(max_len, d_model)           # [5000,512]  
        # 生成pos， [ [0],[1], ··· [max_len-1] ] 的张量
        pos = t.arange(0, max_len).unsqueeze(1)  # [5000,1]，词向量在句子中的位置信息
        m2i = t.arange(0, d_model, 2)       # 生成 2i，i为词向量内维度的位置,i在0-255之间
        log = math.log(10000.0)/d_model     # 生成 log(10000)/d_model
        # [256]  生成位置公式的公共部分，即 10000**(2i/d_model)
        div_term = t.exp(m2i * -log)        

        '''
        pe 对于所有词向量，从词向量的第0个维度开始，间隔为2开始赋值，即偶数位
        pos*div_term:[5000,256]  
        pe:[5000,512]
        '''
        pe[:, 0::2] = t.sin(pos * div_term)
        '''
        pe 对于所有词向量，从词向量的第1个维度开始，间隔为2开始赋值，即奇数位
        pos*div_term:[5000,256]  
        pe:[5000,512]
        '''
        pe[:, 1::2] = t.cos(pos * div_term)
        print((pos * div_term)[1, :].size())
        pe = pe.unsqueeze(0)        # [1,5000,512]         
        self.pe = pe

    '''
    x:   词向量
    '''
    def forward(self, x):
        num = x.size(0)             # 句子中词向量的个数
        x = x + self.pe[:, :num]    # 词向量与位置编码相加
        out = self.dropout(x)
        return out


po = PositionalEncoding(512)
a = t.ones(0,512)
a.unsqueeze(1)
po(a)


