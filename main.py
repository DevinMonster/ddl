from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def solve():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(512, (0.5, 2.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # new_labels, old_labels


if __name__ == '__main__':
    solve()
