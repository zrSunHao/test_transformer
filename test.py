import torch as t

a = [[1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9]]
b = t.tensor(a)
print(b.size())
b = b.unsqueeze(-2)
print(b.size())