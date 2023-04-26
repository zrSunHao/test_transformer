import torch as t

x1 = [[1],[2],[1],[1],[1],]
x1 = t.tensor(x1)

x2 = [1,2,3,4,5,6,7,8,9]
x2 = t.tensor(x2)

a1 = x1*x2
print(a1.size())
print(a1)