import torch
# from matplotlib import pyplot as plt

# load data
words = open('../dataset/names.txt').read().splitlines()
print(words[:10])

# map for word to int coding and decoding
chars = sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}

# Create the training set of all the bigrams
xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

import torch.nn.functional as F

xenc = F.one_hot(xs, num_classes=27).float()
# print(xenc)
# print(xenc.dtype)

W = torch.randn(27, 27)

print(xenc @ W)