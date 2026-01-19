import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import os
from deepfake.deepfakeai.models.base_model import CNN_ViT_LSTM

class Predictor:
    def __init__(self, config: dict):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        models_dir = config["modelsdir"]
        model_name = f"{config['model_name']}.pth"
        model_path = os.path.join(str(models_dir), model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file {model_name} not found in {models_dir},"
                f" Please train or download the model first."
            )
        
        self.model = CNN_ViT_LSTM().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),                 # Standardize input size
            transforms.RandomHorizontalFlip(p=0.5),        # Helps generalization
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])


    def predict_image(self, image_path):
        """ 
        Predict single image (Real / Fake) and return detection scores
        """
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img).unsqueeze(0).unsqueeze(1).to(self.device)  # [B, Seq, C, H, W]

        with torch.no_grad():
            output = self.model(img)
            probs = F.softmax(output, dim=1)[0]  # shape: [2]
            pred = torch.argmax(probs).item()
    
        return {
            "label": "Real" if pred == 0 else "Fake",
            "confidence": probs[pred].item(),
            "scores": {
                "real": probs[0].item(),
                "fake": probs[1].item()
            }
        }


    def predict_video(self, video_path, frame_skip=10, max_frames=20):
        """ 
        Predict video by sampling frames and return detection scores
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_skip == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = self.transform(img).unsqueeze(0).to(self.device)
                frames.append(img)
            count += 1
        cap.release()

        if not frames:
            return {"label": "Error", "confidence": 0.0, "scores": {"real": 0.0, "fake": 0.0}}

        frames = torch.cat(frames, dim=0).unsqueeze(0)  # [1, Seq, C, H, W]

        with torch.no_grad():
            output = self.model(frames)
            probs = F.softmax(output, dim=1)[0]  # shape: [2]
            pred = torch.argmax(probs).item()

        return {
            "label": "Real" if pred == 0 else "Fake",
            "confidence": probs[pred].item(),
            "scores": {
                "real": probs[0].item(),
                "fake": probs[1].item()
            }
        }

            