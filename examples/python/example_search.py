import hnswlib
import torch
import numpy as np
import time

import sys
f = open('a.log', 'w')
sys.stdout = f

dim = 128
num_elements = 128000

k = torch.randn(1, 1, num_elements, dim, device=torch.device("cuda:0"), dtype=torch.float, requires_grad=False)

hnsw_index = hnswlib.Index(space='ip', dim=dim)  # possible options are l2, cosine or ip
hnsw_index.init_index(max_elements=1000000, ef_construction=200, M=90)
hnsw_index.set_ef(100)

k = k.chunk(128, dim=2)

for idx in range(128):
    current_k = k[idx].detach()
    data = current_k[0][0].detach().to(torch.float).cpu().numpy()
    b1 = time.time()
    hnsw_index.add_items(data)
    b2 = time.time()
    print(f"add key latency: {b2-b1} ")

q_full = torch.randn(1, 1, num_elements, dim, device=torch.device("cuda:0"), dtype=torch.float, requires_grad=False)
data = q_full[0][0].detach().to(torch.float).cpu().numpy()
b3 = time.time()
hnsw_index.add_items(data)
b4 = time.time()
print(f"add key latency: {b4-b3} ")

q_de = torch.randn(1, 1, 1, dim, device=torch.device("cuda:0"), dtype=torch.float, requires_grad=False)
data = q_de[0][0].detach().to(torch.float).cpu().numpy()
b5 = time.time()
hnsw_index.add_items(data)
b6 = time.time()
print(f"add key latency: {b6-b5} ")

