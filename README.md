# Hot-Stocks-Predicition-AI-GO-Competition-

## Overview
This repository contains the code for our Statistical Consulting Project, submitted for the AI GO Competition. The project focuses on predicting hot stocks using a dataset with 10,212 features, including brokerage data, financial ratios, and technical indicators.

## Project Structure
### Data Preprocessing:
- data_preprocessing_1.py: Handles initial data preprocessing, including dropping columns with >40% missing values, filling remaining numerical NA with zeros, one-hot encoding categorical variables (e.g., IFRS_DPZ, IFRS_Z), and applying shift-log transformation to high-skew features.
- data_preprocessing_2.py: Enhances the dataset by deriving time-series features, such as averages (1-20 days), log differences (6-day minus current), On-Balance Volume (OBV), and Price Movement Volume (PMV), and applies TAIEX/STK pipelines.

### Modeling:
- stage1_ML_engineer.py: Implements a LightGBM classifier with tuned parameters, 10-fold stratified cross-validation, and threshold optimization (0.1 to 1.0) to maximize F1 score, serving as the baseline model.
- stage2_stacking_LR.py: Builds a stacking ensemble using three LightGBM models (on technical, non-technical, and all features) and a Logistic Regression meta-model, optimized via Optuna for AUPRC.
- stage3_base_XGB.py: Trains XGBoost models on all combinations of three feature categories with 5-fold cross-validation, using a weighted ensemble based on AUPRC.

## Requirements
Install dependencies: pip install -r requirements.txt
