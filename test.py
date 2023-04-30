import torch as t

a = [[1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9]]
b = t.tensor(a)
c = b >= 5
print(c)
d = b + c
print(d)