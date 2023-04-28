import torch as t

def test_yield():
    for i in range(100):
        if i % 2 == 0:
            yield i

for msg in test_yield():
    print(msg)