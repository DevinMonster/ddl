import torch

a = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
a[torch.where(a == 2)] = 3
print(a)


