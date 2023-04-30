import torch as t
from torch.autograd import Variable
import numpy as np

from .mask import subsequent_mask


'''
获取词与此编码的映射
'''
def get_dict(filename, pad=0, eos=2, unk=3):
    token_map = {'PAD':0, '<BOS>': 1, '<EOS>': 2, '<EOS>': 3}
    with open(filename, encoding='utf8') as f:
        for i, line in enumerate(f, start=4):
            keys = line.strip().split()
            token_map[keys[0]] = i
    return token_map



'''
统一batch_size内句子的长度，少的加填充符，即 padding_idx
batch:          句子的集合 [B, num]
padding_idx:    填充的占位符 0
'''
def batch_padding(batch, padding_idx):
    # 获取 batch 中最长的句子
    batch_max = max(batch, key=lambda x: len(x))
    # 获取 batch 中最长句子的长度
    max_len = len(batch_max)
    
    # sentenced
    for sent in batch:
        # 计算需要补充的长度
        padding_len = max_len - len(sent)
        if padding_len:
            placeholder = [padding_idx]*padding_len
            sent.extend(placeholder)
    return batch



'''
训练之前，用掩码处理一批数据，即掩盖填充的占位符
'''
class Batch:

    '''
    src:    经过编码之后的源语言句子集合，  [B, token_num]
    tgt:    经过编码之后的目标语言句子集合，[B, token_num]
    pad:    掩码的填充值，默认为0
    '''
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        '''
        源语言句子的掩码矩阵
        不需要掩盖的位置为True，需要掩盖的位置为False
        [B, token_num] --> [B, 1, token_num]
        '''
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            '''
            每个句子去掉最后一个结束标识符 
            [B, token_num] --> [B, token_num-1]
            '''
            self.tgt = tgt[:,:-1] 
            '''
            每个句子去掉起始位置的一个开始标识符
            [B, token_num] --> [B, token_num-1]
            '''
            self.tgt_y = tgt[:,1:]
            # [B, token_num-1, token_num-1]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            '''
            计算当前batch目标语言的token总数（这里加了结束标识符）
            self.tgt_y != pad
                size:   [4, token_num-1]
                为掩码的位置为False，目标语言token的位置为True
            '''
            self.ntokens = (self.tgt_y != pad).data.sum()

    '''
    创建一个目标语言句子的掩码矩阵
    参数
        tgt:    [B, token_num-1]
        pad:    掩码，默认为0
    返回
        tgt_mask:   [B, token_num-1, token_num-1]
    '''
    @staticmethod 
    def make_std_mask(tgt, pad):
        '''
        目标语言句子的掩码矩阵
        不需要掩盖的位置为True，需要掩盖的位置为False
        [B, token_num-1] --> [B, 1, token_num-1]
        '''
        tgt_mask =  (tgt != pad).unsqueeze(-2)
        '''
        创建维度为[1, token_num-1, token_num-1]的掩码矩阵
        用于掩盖“当前词”后面的词语
        '''
        mask = subsequent_mask(size = tgt.size(-1)).type_as(tgt_mask.data)
        '''
        tgt_mask:   [B, 1, token_num-1]
        mask:       [1, token_num-1, token_num-1]
        tgt_mask = tgt_mask & mask --> [B, token_num-1, token_num-1]
        '''
        tgt_mask = tgt_mask & Variable(mask)
        return tgt_mask



'''
真实的训练数据
'''
def real_data_gen(nbatches, dict_zh_path, dict_en_path, train_zh_path, train_en_path):
    
    dict_zh = get_dict(dict_zh_path)
    dict_en = get_dict(dict_en_path)
    train_zh = open(train_zh_path, encoding='utf8')
    train_en = open(train_en_path, encoding='utf8')

    batch_zh = []
    batch_en = []

    for sent_en, sent_zh in zip(train_en, train_zh):
        # 每个句子添加开头和结尾标识符
        sent_en = '<BOS> {} <EOS>'.format(sent_en.strip())
        sent_zh = '<BOS> {} <EOS>'.format(sent_zh.strip())

        # 将每个句子的每个词映射为对应的编码
        codes_en = []
        for token in sent_en.split():
            code = dict_en[token]
            codes_en.append(code)
        batch_en.append(codes_en)
        codes_zh = []
        for token in sent_zh.split():
            code = dict_zh[token]
            codes_zh.append(code)
        batch_zh.append(codes_zh)

        # 按 batch_size 的大小提供数据据
        if len(batch_en) % nbatches == 0:
            x = batch_zh
            # 统一句子的长度
            src = batch_padding(batch_en, padding_idx = 0)
            src = t.tensor(src, dtype=t.int).long()     #  [B, token_num]
            tgt = batch_padding(batch_zh, padding_idx = 0)
            tgt = t.tensor(tgt, dtype=t.int).long()     #  [B, token_num]
            yield Batch(src, tgt, pad=0)

