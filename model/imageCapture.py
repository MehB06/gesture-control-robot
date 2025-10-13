import cv2
import torch
from torchvision import transforms

import numpy as np
import time

transformation = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load("data/saves/best_checkpoint.pth", map_location=device)

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]


cap = cv2.VideoCapture(0)
last_sample_time = 0
sample_interval = 0.3  # seconds
prediction_text = ""

while True:
    pass 