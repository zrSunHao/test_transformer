import numpy as np
import torch as t


'''
构造解码器所需要的 mask，使后面的单词无法“看到”前面的单词
size:   句子中的单词数量
'''
def subsequent_mask(size):
    attn_shape = (1, size, size)
    matrix = np.ones(attn_shape).astype(np.uint8)
    # np.triu 返回矩阵的上三角部分的值,下半部分为0
    mask = np.triu(matrix)
    mask = t.from_numpy(mask) == 0
    return mask


# 测试
# print(subsequent_mask(5))
# '''
# [[  [False, False, False, False, False],
#     [ True, False, False, False, False],
#     [ True,  True, False, False, False],
#     [ True,  True,  True, False, False],
#     [ True,  True,  True,  True, False]     ]]

# '''