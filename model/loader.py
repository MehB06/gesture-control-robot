import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import BATCH_SIZE, TEST_PERCENT, TRAIN_PERCENT, NUM_WORKERS, DATA_DIR

transformation = transforms.Compose([
    transforms.Resize((200, 200)),       # 200 x 200 Pixels
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize((0.5,), (0.5,)) # Normalize between -1 and 1
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transformation)

trainSize = int(len(dataset)*TRAIN_PERCENT)
testSize = int(len(dataset)*TEST_PERCENT)
valSize = int (len(dataset) - (trainSize + testSize))
trainDataset, valDataset, testDataset = random_split(
    dataset,                                                    # Dataset               
    [trainSize, valSize, testSize], 
    generator=torch.Generator().manual_seed(42)) 


# 4. Create DataLoaders
trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
validationDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testDataLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)