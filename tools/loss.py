import torch as t
import torch.nn as nn
from torch.autograd import Variable


'''
计算损失
'''
class LabelSmoothing(nn.Module):

    '''
    size:   目标语言词表的大小 voc_zh
    padding_idx:    开始标识符所在的位置
    '''
    def __init__(self, voc_size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = voc_size
        self.true_dist = None

    '''
    x:      模型预测的结果      [B*token_num, vocab_zh]
    target: 目标语言的真实结果  [B*token_num]
    '''
    def forward(self, x, target):
        assert x.size(1) == self.size
        #  [B*token_num, vocab_zh]
        true_dist = x.data.clone()

        # self.size - 2 == voc_zh -2 中文表示词表减去开始、结束标识符
        value = self.smoothing / (self.size - 2)    # 这里value为0
        # fill_：将 tensor 中的所有值都填充为指定的 value
        true_dist.fill_(value)
        # 在 true_dist 的 dim 维度，按 index 给定的索引，将值替换为 value
        index = target.data.unsqueeze(1).to(x.device)
        true_dist.scatter_(dim=1, 
                           index=index, # [B*token_num, 1]
                           value=self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = t.nonzero(target.data == self.padding_idx)
        mask = mask.to(x.device)
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
    x:      解码器预测值  [B, token_num, d_model]
    y:      真实值  [B, token_num]
    norm:   batch下目标语言真实token的数量，用于标准化    
    '''
    def __call__(self, x, y, norm):
        '''
        解码器预测完之后，需经过输出层处理，得对应词表得概率
        [B, token_num, d_model] --> [B, token_num, vocab_zh]
        '''
        x = self.generator(x)  
        # 获取词表的大小   
        vocab_zh = x.size(-1) 

        # [B, token_num, vocab_zh] --> [B*token_num, vocab_zh]
        x = x.contiguous().view(-1, vocab_zh)  # [B, num]
        # [B, token_num] --> [B*token_num]
        y = y.contiguous().view(-1)

        loss = self.criterion(x, y)
        loss = loss/norm

        loss.backward()             # 反向传播
        if self.opt is not None:    # 优化器优化
            self.opt.step()
            self.opt.optimizer.zero_grad()
        
        return loss.item() * norm