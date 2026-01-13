Plant Water Needs Classification

This project presents a supervised machine learning model that predicts plant water requirements based on environmental conditions.

The model classifies plants into different water-need categories using structured tabular data.

Problem Statement

Efficient irrigation is essential for sustainable agriculture.
Incorrect watering leads to poor crop health and wastage of water resources.

This project aims to classify plant water needs using measurable environmental parameters.

Dataset Features
Feature	Description
sunlight_hours	Daily sunlight exposure
watering_freq_per_week	Weekly watering frequency
soil_type	Type of soil
water_need	Target label (Low / Medium / High)
Machine Learning Approach

Label Encoding for categorical features

Feature scaling using StandardScaler

SMOTE for class balancing

XGBoost Classifier for training

Model Configuration
Parameter	Value
Algorithm	XGBoost
Train/Test Split	80 / 20
Scaling	StandardScaler
Oversampling	SMOTE
Model Performance
Metric	Value
Accuracy	~0.60 – 0.80
