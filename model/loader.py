import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import BATCH_SIZE, TEST_PERCENT, TRAIN_PERCENT, NUM_WORKERS, DATA_DIR

class DataProcessor:
    def __init__(self):
        
        self.transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((200,200)),
            transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.8,1.2)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.dataset = datasets.ImageFolder(root=DATA_DIR, transform=self.transformation)

        self.trainDataLoader, self.valDataLoader, self.testDataLoader = self.__createDataLoaders()

    def __splitDataset(self):
        trainSize = int(len(self.dataset)*TRAIN_PERCENT)
        testSize = int(len(self.dataset)*TEST_PERCENT)
        valSize = int (len(self.dataset) - (trainSize + testSize))

        return random_split(self.dataset,
                            [trainSize, valSize, testSize],
                            generator=torch.Generator().manual_seed(42)) 

    def __createDataLoaders(self):
        trainDataset, valDataset, testDataset = self.__splitDataset()

        trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        testDataLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        return trainDataLoader, valDataLoader, testDataLoader
    
    def getDataLoaders(self):
        return self.trainDataLoader, self.valDataLoader, self.testDataLoader