import torch
from torch import nn

print(torch.version.cuda)
print(torch.__version__)

B = 4
T = 8
C = 32
head_size = 16

x = torch.randn(B, T, C)

key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
k = key(x)
q = query(x)