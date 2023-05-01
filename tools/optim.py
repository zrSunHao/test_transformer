import torch as t
import torch.optim as optim

'''
优化器
'''
class NoamOpt:

    '''
    Optim wrapper that implements rate.
        model_size: d_model
        factor:
        warmup:
        optimizer:
    '''
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0      # 学习率

    '''
    Update parameters and rate  
    '''
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    '''
    Implement `lrate` above
    '''
    def rate(self, step=None):
        if step is None:
            step = self._step
        min_step = min(step**(-0.5), step*self.warmup**(-1.5))
        out = self.factor * (self.model_size**(-0.5) * min_step)
        return out
    

'''
获取优化器
'''
def get_std_opt(model):
    # model.src_embed[0] 即 Embeddings
    model_size = model.src_embed[0].d_model
    factor = 2
    warmup = 4000
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    opt = NoamOpt(model_size, factor, warmup, optimizer)
    return opt