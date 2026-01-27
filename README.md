# Multi-frequency street-level urban noise modeling and mapping through street view and remote sensing image fusion - DINOv3

A deep learning-based system for predicting environmental noise levels from street view images using DINOv3 features and machine learning models.

<span style="color:red;">More dataset and code will coming soon!</span>

## Overview

This project predicts noise pollution levels from street view imagery by:
1. Extracting visual features using Facebook's DINOv3 vision transformer
2. Processing features through trained preprocessing pipelines
3. Predicting noise levels using various machine learning models (KNN, Random Forest, XGBoost, etc.)

## Result
**Buffer** 15m->25m->50m->100m
<img width="865" height="514" alt="image" src="https://github.com/user-attachments/assets/831c96b8-d55c-49bf-b125-d2ee8556e54e" />
<img width="865" height="517" alt="image" src="https://github.com/user-attachments/assets/69862240-07d9-4b6c-9984-623215dbc152" />
<img width="865" height="526" alt="image" src="https://github.com/user-attachments/assets/83a791d8-c6a0-4fc6-92e0-f1c90894715e" />
<img width="865" height="527" alt="image" src="https://github.com/user-attachments/assets/74063e75-5dfc-474e-a8d7-c3d75227a4ec" />

## Project Structure

NoiseMap/
├── predict_noise.py              # Main prediction script
├── Train and Experiment.ipynb    # Model training notebook
├── README.md                      # This file
│
├── test_images/                   # Input images for prediction (create this folder)
│   └── [your images here]
│
├── predictions.csv                # Output predictions (generated)
│
└── saved_models/                  # Pre-trained models
    ├── predictor_complete.pkl    # Complete predictor (optional)
    │
    ├── Mean_Noise/               # Mean noise level models
    ├── Low_Freq/                 # Low frequency noise models
    ├── Mid_Freq/                 # Mid frequency noise models
    └── High_Freq/                # High frequency noise models
        │
        ├── street/               # Street view-based models
        ├── remote/               # Remote sensing-based models
        └── fusion/               # Multi-modal fusion models
            │
            ├── KNN_model.pkl
            ├── KNN_preprocessors.pkl
            ├── KNN_info.pkl
            │
            ├── Random Forest_model.pkl
            ├── Random Forest_preprocessors.pkl
            ├── Random Forest_info.pkl
            │
            └── [other models...]


## Prediction Categories

### Noise Types
- Mean_Noise: Overall average noise level
- Low_Freq: Low frequency noise (e.g., traffic rumble)
- Mid_Freq: Mid frequency noise (e.g., conversation)
- High_Freq: High frequency noise (e.g., brakes, horns)

### Data Sources
- street: Street view image features only
- remote: Remote sensing data only
- fusion: Combined street view + remote sensing

### Available Models
- KNN: K-Nearest Neighbors
- Random Forest: Random Forest Regressor
- Gradient Boosting: Gradient Boosting Regressor
- XGBoost: XGBoost Regressor
- LightGBM: LightGBM Regressor
- SVR: Support Vector Regressor
- Lasso: Lasso Regression
- Stacking: Stacked ensemble model
- Voting: Voting ensemble model

## Quick Start

### Model

https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m
https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

### Prerequisites

pip install torch torchvision transformers
pip install numpy pandas scikit-learn joblib pillow tqdm
pip install xgboost lightgbm
pip install huggingface_hub

### Basic Usage

1. Prepare your images:

mkdir test_images
# Copy your street view images to test_images/

2. Configure the prediction script:

Edit predict_noise.py to set your preferences:

IMAGE_FOLDER = "test_images"
OUTPUT_FILE = "predictions.csv"
MODEL_PATH = "saved_models/Mean_Noise/street/KNN_model.pkl"
PREPROCESSOR_PATH = "saved_models/Mean_Noise/street/KNN_preprocessors.pkl"
HF_TOKEN = 'your_huggingface_token_here'

3. Run prediction:

python predict_noise.py

4. View results:

image_name: Image filename
mean_noise_dB: Predicted noise level in decibels
noise_level: Category (Quiet/Rather Quiet/Moderate/Rather Noisy/Noisy)
📊 Noise Level Categories
Category	dB Range	Description
Quiet	< 50 dB	Very peaceful environment
Rather Quiet	50-60 dB	Residential areas
Moderate	60-70 dB	Normal urban areas
Rather Noisy	70-80 dB	Busy streets
Noisy	> 80 dB	Heavy traffic/industrial

## Advanced Configuration

### Switching Models

To use a different model, update the paths in predict_noise.py:

# Example: Using XGBoost for High Frequency noise with fusion data
MODEL_PATH = "saved_models/High_Freq/fusion/XGBoost_model.pkl"
PREPROCESSOR_PATH = "saved_models/High_Freq/fusion/XGBoost_preprocessors.pkl"

### Batch Processing

The script automatically processes all images in the specified folder:
- Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif
- Images are processed sequentially with progress tracking
- Failed images are skipped with error messages

## Model Training

To train your own models, use the Jupyter notebook:

jupyter notebook "Train and Experiment.ipynb"

The notebook includes:
- Data loading and preprocessing
- Feature extraction with DINOv3
- Model training and hyperparameter tuning
- Cross-validation and evaluation
- Model saving and export

## Example Output

Using device: cuda

Loading DINOv3 model: facebook/dinov3-vitb16-pretrain-lvd1689m
✓ DINOv3 model loaded successfully

Loading KNN model and preprocessors...
✓ KNN model loaded successfully

Found 150 images
============================================================

Extracting DINOv3 features...
Processing images: 100%|████████████| 150/150 [02:15<00:00,  1.11it/s]
Extracted feature dimensions: (150, 768)

Applying preprocessing pipeline:
  Initial dimensions: (150, 768)
  [1/3] Scaling (768-dim)... → (150, 768)
  [2/3] Feature selection (300-dim)... → (150, 300)
  [3/3] PCA reduction (148-dim)... → (150, 148)
  Final dimensions: (150, 148)

Executing prediction...

============================================================
Prediction completed!
============================================================

Statistics:
  Successfully processed: 150 images
  Mean noise: 65.34 dB
  Std dev:    8.12 dB
  Min value:  48.22 dB
  Max value:  82.45 dB
  Median:     64.80 dB

Noise level distribution:
  Quiet        :  12 images (  8.0%)
  Rather Quiet :  35 images ( 23.3%)
  Moderate     :  58 images ( 38.7%)
  Rather Noisy :  32 images ( 21.3%)
  Noisy        :  13 images (  8.7%)

Full results saved to: predictions.csv

## HuggingFace Token

To use DINOv3, you need a HuggingFace token:

1. Create account at https://huggingface.co
2. Go to Settings → Access Tokens
3. Create a new token with read permissions
4. Add to predict_noise.py:

HF_TOKEN = 'hf_YourTokenHere'

## Technical Details

### Feature Extraction Pipeline
1. DINOv3 Encoding: Images → 768-dimensional feature vectors
2. Standardization: Z-score normalization
3. Feature Selection: Select 300 most important features
4. PCA: Reduce to 148 dimensions

### Model Files
Each model configuration includes three files:
- *_model.pkl: Trained model weights
- *_preprocessors.pkl: Feature preprocessing pipeline
- *_info.pkl: Training metadata and performance metrics

## Troubleshooting

### CUDA Out of Memory
# Switch to CPU
self.device = torch.device('cpu')

### Model Loading Error
- Verify file paths exist
- Check pickle compatibility with current scikit-learn version
- Ensure all required libraries are installed

### Feature Dimension Mismatch
- The preprocessing order is critical: Scaling → Selection → PCA
- Check preprocessor inspection output for expected dimensions

## Citation

If you use this system in your research, please cite:

Yan Zhang, Entong Ke, Mei-Po Kwan, Libo Fang, Mingxiao Li,
Multi-frequency street-level urban noise modeling and mapping through street view and remote sensing image fusion,
Computers, Environment and Urban Systems,
Volume 126,
2026,
102401,
ISSN 0198-9715,
https://doi.org/10.1016/j.compenvurbsys.2026.102401.
(https://www.sciencedirect.com/science/article/pii/S0198971526000037)
Abstract: Urban noise pollution has become the third most significant environmental health threat following air and water pollution, while traditional noise modeling methods suffer from limitations including high costs, limited coverage, and an exclusive focus on total decibel values while neglecting frequency characteristics. This study proposes a method that combines street view imagery (SVI) and remote sensing imagery (RSI) to achieve precise modeling and mapping of multi-frequency noise exposure at the urban street scale. Using Xiangzhou District, Zhuhai City as a case study, we utilized approximately 6000 street view images and corresponding remote sensing images, and recorded 35,276 street noise audios containing 23 frequency bands (100 Hz-16,000 Hz) through volunteer cycling surveys. A multi-source fusion model was constructed based on a pre-trained vision transformer architecture, with 923 valid street noise-image paired samples used for training and validation. The sensitivity results demonstrate that: (1) the proposed multimodal fusion model achieves high predictive accuracy, with R2 values for dBA prediction ranging from 0.417 to 0.649, with particularly higher accuracy observed for mid-frequency noise prediction; (2) 50-m resolution street-scale multi-frequency soundscape maps were successfully generated, providing scientific evidence for refined urban noise management; (3) explainable machine learning models revealed that buildings, roads, sidewalks, and terrain visual elements are the four most important factors affecting noise prediction, with road width showing a positive association with street noise levels. This study not only fills the gap in urban noise frequency characteristics research but also provides new methodological support for precise street-level noise pollution modeling and health-oriented urban planning. The source code is available at https://github.com/giserzy/NoisePrediction.
Keywords: Noise exposure; Information fusion; Street view image; Noise modeling; Sensing bias

## Contact

Email: yanzhang@cuhk.edu.hk

Last Updated: January 2026
