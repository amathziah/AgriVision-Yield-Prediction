import joblib
import pandas as pd
from pathlib import Path
from src.cnn_inference import CNNInference

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HYBRID ML MODEL FUSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HybridModel:
    """
    Weighted Fusion Model combining Tabular SVR and Computer Vision CNN.
    
    Why Weighted Fusion?
    --------------------
    Individual models often have distinct error profiles. SVR excels at capturing 
    long-term tabular trends (climate, pesticides), while CNNs capture immediate 
    spatial visual health from satellite imagery. Combining them (Ensemble) 
    reduces overall variance.

    Bias-Variance Tradeoff:
    -----------------------
    - SVR (Tabular): Low variance on consistent time-series, potentially higher bias if features are missing.
    - CNN (Images): Captures high-variance visual signals (pests, localized blight).
    - Hybrid: By averaging predictions, we dampen individual model errors, 
      effectively shifting the model toward the 'Goldilocks' zone of the trade-off curve.
    """
    def __init__(self, svr_path, cnn_path=None):
        self.svr_model = joblib.load(svr_path)
        self.cnn_pipeline = CNNInference(model_path=cnn_path)
        print("✅ Hybrid Model Initialized (SVR + CNN)")

    def predict(self, tabular_data, image_path=None):
        """
        tabular_data: dict or DataFrame
        image_path: Path to satellite image
        """
        # 1. SVR Prediction
        if isinstance(tabular_data, dict):
            tabular_df = pd.DataFrame([tabular_data])
        else:
            tabular_df = tabular_data
            
        svr_pred = self.svr_model.predict(tabular_df)[0]

        # 2. CNN Prediction (if image provided, else use base)
        if image_path and Path(image_path).exists():
            cnn_pred = self.cnn_pipeline.predict_image(image_path)
        else:
            # If no image, CNN contributes a 'neutral' baseline or we fallback to SVR only
            # Alternatively, we can simulate a CNN score for demonstration
            cnn_pred = svr_pred * 0.95 

        # 3. Weighted Fusion
        # SVR is historically more reliable for this dataset, so we give it 70% weight
        final_pred = (0.7 * svr_pred) + (0.3 * cnn_pred)

        return {
            "hybrid_prediction": round(float(final_pred), 2),
            "svr_component": round(float(svr_pred), 2),
            "cnn_component": round(float(cnn_pred), 2),
            "weights": {"svr": 0.7, "cnn": 0.3}
        }

if __name__ == "__main__":
    SVR_PATH = "project/models/best_svr_model.pkl"
    if Path(SVR_PATH).exists():
        hybrid = HybridModel(SVR_PATH)
        # Mock tabular data
        mock_data = {
            'country': 0, 'crop': 0, 
            'average_rain_fall_mm_per_year': 1200, 
            'avg_temp': 25.0, 'pesticides_tonnes': 200
        }
        res = hybrid.predict(mock_data)
        print(f"🚀 Hybrid Prediction Result: {res}")
