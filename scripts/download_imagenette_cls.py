import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
import urllib.request
import tarfile
import os

# Create directory if it doesn't exist
if not os.path.exists('./data/imagenette2-160'):
    # Download Imagenette-160
    url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
    urllib.request.urlretrieve(url, './data/imagenette2-160.tgz')
    
    # Extract the dataset
    with tarfile.open('./data/imagenette2-160.tgz', 'r:gz') as tar:
        tar.extractall('./data')

# Load training data
imagenette_train = datasets.ImageFolder(
    root='./data/imagenette2-160/train',
    transform=None  # You can add transforms here if needed
)

# Load validation data
imagenette_val = datasets.ImageFolder(
    root='./data/imagenette2-160/val',
    transform=None  # You can add transforms here if needed
)

# Create subsets with desired sizes
imagenette_train_subset = Subset(imagenette_train, list(range(240)))
imagenette_val_subset = Subset(imagenette_val, list(range(60)))

# Print dataset sizes
print(f"Training subset size: {len(imagenette_train_subset)}")
print(f"Validation subset size: {len(imagenette_val_subset)}")