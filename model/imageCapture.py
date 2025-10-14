import cv2
import torch
import time
from cnn import CNN 
from torchvision import transforms
from PIL import Image

# Model Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN()
model.to(device)

checkpoint = torch.load("data/saves/last_checkpoint.pth", map_location=device)  # adjust path to your file
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Classes
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
          "del", "nothing", "space"]


# Process Frame
transformation = transforms.Compose([
    transforms.Resize((200, 200)),       # 200 x 200 Pixels
    transforms.ToTensor(),               # Convert to tensor
    transforms.Normalize((0.5,), (0.5,)) # Normalize between -1 and 1
])

# Camara capture 
videoCam = cv2.VideoCapture(0)
lastSampleTime = 0
sampleRate = 0.3  # seconds
prediction = ""

while True:
    ret, frame = videoCam.read()
    if not ret:
        break

    if (time.time() - lastSampleTime) > sampleRate:
        lastSampleTime - time.time()

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputTensor = transformation(pil_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(inputTensor)
            predictionIndex = torch.argmax(outputs, dim=1).item()
            prediction = labels[predictionIndex]

    # Image Feedback
    cv2.putText(frame, f"Prediction: {prediction}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


videoCam.release()
cv2.destroyAllWindows()
