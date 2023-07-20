from models import HierarchicalEmbedding
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor
import torch
from torchvision.datasets.cifar import CIFAR100
from torch.utils.data import DataLoader, random_split
from utils.scheduler import StepLR
from utils import CrossEntropyLoss
from tqdm import tqdm
from torch.cuda.amp import autocast

epochs = 200
batch_size = 128
lr = 0.1
# train/valid split ratio in (0, 1)
ratio = 0.8
device = torch.device(f"cuda:{0}")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = Compose([
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std)
])
test_transform = Compose([
    ToTensor(),
    Normalize(mean, std)
])

total_ds = CIFAR100("./data", transform=train_transform)
test_ds = CIFAR100("./data", train=False, transform=test_transform)
train_size = int(ratio * len(total_ds))
valid_size = len(total_ds) - train_size
train_ds, valid_ds = random_split(total_ds, [train_size, valid_size])
# loader
train_loader = DataLoader(train_ds, batch_size, True, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size)
test_loader = DataLoader(test_ds, batch_size)

model = HierarchicalEmbedding(img_size=32, patch_size=4, num_classes=100).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr, 0.9)
T = epochs * train_size // batch_size
scheduler = StepLR(optimizer, 40, 0.4)
loss = CrossEntropyLoss()

best = 0.
best_state = None
for epoch in range(1, epochs + 1):
    print(f"epoch {epoch} begin!")
    # train
    losses = 0
    correct = 0
    batches = 0
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)
        with autocast():
            f, y_hat = model(x)
            l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        losses += l.item()
        correct += (torch.argmax(y_hat, dim=1) == y).sum().item()
        batches += 1
    scheduler.step()
    losses /= batches
    acc = correct / valid_size
    print(f"epoch:{epoch}'s train loss is {losses:.4f}, accuracy is {acc:.3f}, learning rate {scheduler.get_last_lr()}")

    # valid
    losses = 0
    correct = 0
    batches = 0
    for x, y in tqdm(valid_loader):
        x = x.to(device)
        y = y.to(device)
        with autocast():
            f, y_hat = model(x)
            l = loss(y_hat, y)
        losses += l.item()
        correct += (torch.argmax(y_hat, dim=1) == y).sum().item()
        batches += 1
    scheduler.step()
    losses /= batches
    acc = correct / valid_size
    if acc > best:
        best = acc
        best_state = model.state_dict()
        print(f"BEST ACC:{acc:.3f}")
    print(f"epoch:{epoch}'s valid loss is:{losses:.4f}, accuracy is {acc:.3f}")
