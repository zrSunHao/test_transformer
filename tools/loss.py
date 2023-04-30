import torch as t
import torch.nn as nn
from torch.autograd import Variable


'''
计算损失
'''
class LabelSmoothing(nn.Module):

    '''
    size:   d_model
    padding_idx:    填充标识符，默认为 0
    '''
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    '''
    x:      源语言
    target: 目标语言
    '''
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()

        value = self.smoothing / (self.size - 2)
        # fill_：将 tensor 中的所有值都填充为指定的 value
        true_dist.fill_(value)
        # 用来编码 one hot
        true_dist.scatter_(dim=1, 
                           index=target.data.unsqueeze(1), 
                           src=self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = t.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))



'''
简易的损失函数
'''
class SimpleLossCompute:

    '''
    generator:  解码器的输出层
    criterion:  计算损失得函数
    opt:        优化器
    '''
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    
    '''
    x:      模型预测值  [B, num, tgt_vocab]
    y:      真实值  [B, num-1]
    norm:   标准化    
    '''
    def __call__(self, x, y, norm):
        x = self.generator(x)       # 解码器预测完之后，需经过输出层处理，得对应词表得概率
        v = x.size(-1)              # [B, num]

        x = x.contiguous().view(-1, v)  # [B, num]
        y = y.contiguous().view(-1)     # [B * num-1] TODO 未知维度
        loss = self.criterion(x, y)/norm

        loss.backward()             # 反向传播
        if self.opt is not None:    # 优化器优化
            self.opt.step()
            self.optoptimizer.zero_grad()
        
        return loss.item() * norm