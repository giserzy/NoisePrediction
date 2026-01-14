"""
Street View Image Noise Prediction System - DINOv3 Version (Sequence Corrected)
Using trained KNN model for prediction
Configuration: Mean_Noise + street + KNN
Feature Extraction: DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)
"""

import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration Section ====================
IMAGE_FOLDER = "test_images"              # Folder containing images to predict
OUTPUT_FILE = "predictions.csv"           # Path to save prediction results
MODEL_PATH = "saved_models/Mean_Noise/street/KNN_model.pkl"
PREPROCESSOR_PATH = "saved_models/Mean_Noise/street/KNN_preprocessors.pkl"
DINOV3_MODEL = 'facebook/dinov3-vitb16-pretrain-lvd1689m'  # DINOv3 model
HF_TOKEN = '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'  # Your HuggingFace token
# ================================================================


class DINOv3NoisePredictor:
    """Noise predictor using DINOv3 features"""
    
    def __init__(self, model_path, preprocessor_path, dinov3_model, hf_token=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 1. Login to HuggingFace (if needed)
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        
        # 2. Load DINOv3 model
        print(f"\nLoading DINOv3 model: {dinov3_model}")
        self.processor = AutoImageProcessor.from_pretrained(dinov3_model)
        self.dinov3_model = AutoModel.from_pretrained(dinov3_model, device_map="auto")
        self.dinov3_model.eval()
        print("✓ DINOv3 model loaded successfully")
        
        # 3. Load trained KNN model
        print("\nLoading KNN model and preprocessors...")
        self.knn_model = joblib.load(model_path)
        self.preprocessors = joblib.load(preprocessor_path)
        print("✓ KNN model loaded successfully")
        
        # 4. Inspect preprocessor structure
        self._inspect_preprocessors()
    
    def _inspect_preprocessors(self):
        """Inspect and display detailed preprocessor information"""
        print("\nPreprocessor Details:")
        print("-" * 60)
        
        if isinstance(self.preprocessors, dict):
            for key, value in self.preprocessors.items():
                if value is not None:
                    print(f"  {key}: {type(value).__name__}")
                    
                    # Display feature selector information
                    if key == 'feature_selector' and hasattr(value, 'get_support'):
                        n_selected = value.get_support().sum()
                        n_total = len(value.get_support())
                        print(f"    → Selected {n_selected}/{n_total} features")
                    
                    # Display scaler information
                    elif key == 'scaler' and hasattr(value, 'n_features_in_'):
                        print(f"    → Expected input: {value.n_features_in_} features")
                    
                    # Display PCA information
                    elif key == 'pca_transformer' and hasattr(value, 'n_components_'):
                        print(f"    → Reduced to: {value.n_components_} dimensions")
                else:
                    print(f"  {key}: None (skipped)")
        else:
            print(f"  Type: {type(self.preprocessors).__name__}")
        
        print("-" * 60)
        print("\nInferred preprocessing order: Scaling(768) → Feature Selection(300) → PCA(148)")
        print("-" * 60)
    
    def extract_features(self, image_path):
        """Extract features from image using DINOv3"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.dinov3_model(**inputs)
            feature = outputs.pooler_output.cpu().numpy().flatten()
        
        return feature
    
    def preprocess_features(self, features):
        """
        Apply preprocessing pipeline from training
        Correct order: Scaling(768-dim) → Feature Selection(300-dim) → PCA(148-dim)
        """
        X = features.copy()
        print(f"  Initial dimensions: {X.shape}")
        
        if isinstance(self.preprocessors, dict):
            # Step 1: Scaling (on original 768-dim features)
            if 'scaler' in self.preprocessors and self.preprocessors['scaler'] is not None:
                print(f"  [1/3] Scaling (768-dim)...", end='')
                X = self.preprocessors['scaler'].transform(X)
                print(f" → {X.shape}")
            
            # Step 2: Feature selection (reduce from 768 to 300)
            if 'feature_selector' in self.preprocessors and self.preprocessors['feature_selector'] is not None:
                print(f"  [2/3] Feature selection (300-dim)...", end='')
                X = self.preprocessors['feature_selector'].transform(X)
                print(f" → {X.shape}")
            
            # Step 3: PCA dimensionality reduction (reduce from 300 to 148)
            if 'pca_transformer' in self.preprocessors and self.preprocessors['pca_transformer'] is not None:
                print(f"  [3/3] PCA reduction (148-dim)...", end='')
                X = self.preprocessors['pca_transformer'].transform(X)
                print(f" → {X.shape}")
            
            print(f"  Final dimensions: {X.shape}")
        else:
            # Single preprocessor
            X = self.preprocessors.transform(X)
            print(f"  Preprocessed dimensions: {X.shape}")
        
        return X
    
    def categorize_noise(self, noise_value):
        """Categorize noise level"""
        if noise_value < 50:
            return 'Quiet'
        elif noise_value < 60:
            return 'Rather Quiet'
        elif noise_value < 70:
            return 'Moderate'
        elif noise_value < 80:
            return 'Rather Noisy'
        else:
            return 'Noisy'
    
    def predict(self, image_folder, output_file):
        """Batch predict noise levels for images"""
        
        # 1. Find all images
        image_folder = Path(image_folder)
        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            image_paths.extend(list(image_folder.glob(f'*{ext}')))
            image_paths.extend(list(image_folder.glob(f'*{ext.upper()}')))
        
        image_paths = sorted(image_paths)
        
        if not image_paths:
            raise ValueError(f"No image files found in {image_folder}")
        
        print(f"\nFound {len(image_paths)} images")
        print("="*60)
        
        # 2. Extract features
        print("\nExtracting DINOv3 features...")
        all_features = []
        image_names = []
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                features = self.extract_features(img_path)
                all_features.append(features)
                image_names.append(img_path.stem)
            except Exception as e:
                print(f"\n  ✗ Skipping {img_path.name}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No features were successfully extracted")
        
        # Convert to array
        features_array = np.vstack(all_features)
        print(f"Extracted feature dimensions: {features_array.shape}")
        
        # 3. Preprocess
        print("\nApplying preprocessing pipeline:")
        features_processed = self.preprocess_features(features_array)
        
        # 4. Predict
        print("\nExecuting prediction...")
        predictions = self.knn_model.predict(features_processed)
        
        # Apply inverse transform if target transformer exists
        if isinstance(self.preprocessors, dict):
            if 'target_transformer' in self.preprocessors and self.preprocessors['target_transformer'] is not None:
                print("  Applying target inverse transform...")
                predictions = self.preprocessors['target_transformer'].inverse_transform(
                    predictions.reshape(-1, 1)
                ).flatten()
        
        # 5. Organize results
        results_df = pd.DataFrame({
            'image_name': image_names,
            'mean_noise_dB': predictions.round(2),
            'noise_level': [self.categorize_noise(p) for p in predictions]
        })
        
        # 6. Save results
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 7. Display results
        print("\n" + "="*60)
        print("Prediction completed!")
        print("="*60)
        print(f"\nStatistics:")
        print(f"  Successfully processed: {len(predictions)} images")
        print(f"  Mean noise: {predictions.mean():.2f} dB")
        print(f"  Std dev:    {predictions.std():.2f} dB")
        print(f"  Min value:  {predictions.min():.2f} dB")
        print(f"  Max value:  {predictions.max():.2f} dB")
        print(f"  Median:     {np.median(predictions):.2f} dB")
        
        # Noise level distribution
        print(f"\nNoise level distribution:")
        level_counts = results_df['noise_level'].value_counts()
        for level in ['Quiet', 'Rather Quiet', 'Moderate', 'Rather Noisy', 'Noisy']:
            if level in level_counts.index:
                count = level_counts[level]
                percentage = (count / len(results_df)) * 100
                print(f"  {level:13s}: {count:3d} images ({percentage:5.1f}%)")
        
        print(f"\nResult preview (first 10 entries):")
        print(results_df.head(10).to_string(index=False))
        
        print(f"\nFull results saved to: {output_file}")
        
        return results_df


def main():
    """Main function"""
    try:
        print("="*60)
        print("Street View Noise Prediction System - DINOv3 Version")
        print("="*60)
        
        # Create predictor
        predictor = DINOv3NoisePredictor(
            model_path=MODEL_PATH,
            preprocessor_path=PREPROCESSOR_PATH,
            dinov3_model=DINOV3_MODEL,
            hf_token=HF_TOKEN
        )
        
        # Execute prediction
        results = predictor.predict(IMAGE_FOLDER, OUTPUT_FILE)
        
        print("\n" + "="*60)
        print("✓ All tasks completed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
