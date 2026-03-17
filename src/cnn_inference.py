import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.cnn_model import SatelliteCNN
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CNN INFERENCE PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CNNInference:
    def __init__(self, model_path=None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = SatelliteCNN(num_classes=2).to(self.device)
        self.model.eval()

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded CNN weights from {model_path}")
        else:
            print("⚠️ CNN weights not found. Using initialized (random/simulated) weights for inference.")

        # Match the preprocessing used in image_pipeline.py
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_image(self, image_path):
        """
        Runs inference on a single image.
        Returns a numerical score normalized to 0-100 range (tonnes/ha proxy).
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                # For this hybrid pipeline, we treat the logit/softmax as a 'health' or 'yield' proxy
                # Let's use the softmax of the first class as a proxy for yield potential
                probs = torch.softmax(outputs, dim=1)
                score = probs[0][0].item() # Higher prob of class 0 = higher yield proxy

            # Normalize output to a realistic yield scale (e.g., 0 to 60 t/ha)
            # This is a heuristic normalization for the hybrid baseline
            normalized_yield = score * 60.0 
            
            return round(normalized_yield, 4)
        except Exception as e:
            print(f"❌ CNN Inference Error: {e}")
            return 0.0

if __name__ == "__main__":
    # Test block
    inference = CNNInference()
    # Mock prediction if images aren't available
    test_score = inference.predict_image("project/data/images/test_sample.jpg") if Path("project/data/images/test_sample.jpg").exists() else 35.5
    print(f"📊 Simulated Image Yield Proxy: {test_score} t/ha")
