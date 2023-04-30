import torch as t

src = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
index = t.tensor([[0], [0], [1]], dtype=t.int64)
src.scatter_(dim=1, index=index, value=1.0)
print(src)