import cv2
import torch
import time
from torchvision import transforms
from PIL import Image

from .config import LAST_CHECKPOINT
from .cnn import CNN 

# Model Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ASLDetector():
    def __init__(self,save_dir):
        # Loading Model
        self.__model = self.__getModel(save_dir)

        # Classification lables
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
          "del", "nothing", "space"]
        
        # Process Frame
        self.transformation = transforms.Compose([
            transforms.Resize((200,200)),
            transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.8,1.2)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Camera Capture
        self.videoCam = cv2.VideoCapture(0)
        self.SAMPLE_RATE = 0.3

        self.run()
    
    def run(self):
        lastSampleTime = 0
        prediction = ""
        while True:
            ret, frame = self.videoCam.read()
            if not ret:
                break

            if (time.time() - lastSampleTime) > self.SAMPLE_RATE:
                lastSampleTime - time.time()

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputTensor = self.transformation(pil_frame).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = self.__model(inputTensor)
                    predictionIndex = torch.argmax(outputs, dim=1).item()
                    prediction = self.labels[predictionIndex]

            # Image Feedback
            cv2.putText(frame, f"Prediction: {prediction}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("ASL Live Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.videoCam.release()
        cv2.destroyAllWindows()

            
    def __getModel(self,save_dir):
        model = CNN()
        model.to(device)

        checkpoint_path = save_dir / LAST_CHECKPOINT
        checkpoint = torch.load(checkpoint_path, map_location=device)  # adjust path to your file
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model