import torch.nn as nn


'''
计算损失
'''
class LabelSmoothing(nn.Module):
    def __init__(self) -> None:
        super().__init__()



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