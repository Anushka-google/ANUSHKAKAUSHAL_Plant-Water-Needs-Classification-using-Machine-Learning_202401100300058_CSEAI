# Plant Water Needs Classification

A supervised machine learning model to classify plant water requirements using environmental features.

## Features
- sunlight_hours
- watering_freq_per_week
- soil_type

## Model
- Random Forest Classifier
- Train/Test split: 80/20
- Scaling: StandardScaler

## Results
| Metric | Value |
|-------|------|
| Accuracy | 0.25 |

## How to Run
pip install -r requirements.txt  
python src/train.py
