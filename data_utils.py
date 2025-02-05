# data_utils.py
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_random_balanced_mnist(num_samples=4000, transform=None, seed=42):
    """
    Randomly loads a balanced sample (by class) of num_samples MNIST images.
    Assumes that there are 10 classes, so approximately num_samples//10 images per class.
    """
    random.seed(seed)
    ds = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    indices_by_class = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(ds):
        if len(indices_by_class[label]) < (num_samples // 10):
            indices_by_class[label].append(idx)
        if all(len(v) >= (num_samples // 10) for v in indices_by_class.values()):
            break
    indices = []
    for i in range(10):
        indices.extend(indices_by_class[i])
    random.shuffle(indices)
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=num_samples, shuffle=False)
    X, y = next(iter(loader))
    return X.numpy(), y.numpy()

def get_transform_source():
    """
    Transformation for the source domain (original MNIST images).
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])