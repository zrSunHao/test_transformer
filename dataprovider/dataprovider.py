import torch as t
from torch.autograd import Variable

from .mask import subsequent_mask


'''
获取词与此编码的映射
'''
def get_dict(filename, pad=0, eos=2, unk=3):
    token_map = {'PAD':0, '<BOS>': 1, '<EOS>': 2, '<EOS>': 3}
    with open(filename) as f:
        for i, line in enumerate(f, start=4):
            keys = line.strip().split()
            token_map[keys[0]] = i
    return token_map



'''
统一句子的长度，少的加填充符，即 padding_idx
batch:          句子的集合
padding_idx:    填充的占位符
'''
def batch_padding(batch, padding_idx):
    # 获取 batch 中最长句子的长度
    batch_max = max(batch, key=lambda x: len(x))
    max_len = len(batch_max)
    # sentence
    for sent in batch:
        padding_len = max_len - len(sent)
        if padding_len:
            placeholder = [padding_idx]*padding_len
            sent.extend(placeholder)
    return batch



'''
训练之前，用掩码处理一批数据，即掩盖填充的占位符
pad:    掩码的填充值，默认为0
src:    [B, num]
tgt:    [B, num]
'''
class Batch:

    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # [B, num, 1]
        if tgt is None:
            self.tgt = tgt[:,:-1]   # 每个句子去掉最后一个结束标识符 [B, num-1]
            self.tgt_y = tgt[:,1:]  # 每个句子去掉起始位置的一个开始标识符 [B, num-1]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()   # [B, 1]

    # 创建一个掩码来隐藏填充和未来的单词
    @staticmethod 
    def make_std_mask(tgt, pad):
        tgt_mask =  (tgt != pad).unsqueeze(-2)  # [B, num-1, 1]
        mask = subsequent_mask(size = tgt.size(-1)).type_as(tgt_mask.data)
        tgt_mask = tgt_mask & Variable(mask)    # [B, num-1, 1]
        return tgt_mask



'''
真实的训练数据
'''
def real_data_gen(nbatches, dict_zh_path, dict_en_path, train_zh_path, train_en_path):
    
    dict_zh = get_dict(dict_zh_path)
    dict_en = get_dict(dict_en_path)
    train_zh = open(train_zh_path)
    train_en = open(train_en_path)

    batch_zh = []
    batch_en = []

    for sent_en, sent_zh in zip(train_en, train_zh):
        # 每个句子添加开头和结尾标识符
        sent_en = '<BOS> {} <EOS>'.format(sent_en.strip())
        sent_zh = '<BOS> {} <EOS>'.format(sent_zh.strip())
        # 将每个句子的每个词映射为对应的编码
        batch_en.append( [dict_en[token]] for token in sent_en.split())
        batch_zh.append( [dict_zh[token]] for token in sent_zh.split())

        # 按 batch_size 的大小提供数据据
        if len(batch_en) % nbatches == 0:
            src = batch_padding(batch_en, padding_idx = 0)
            src = t.tensor(src, dtype=t.int).long()     #  [B, num]
            tgt = batch_padding(batch_zh, padding_idx = 0)
            tgt = t.tensor(tgt, dtype=t.int).long()     #  [B, num]
            yield Batch(src, tgt, 0)

