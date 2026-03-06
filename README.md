# House Price Prediction - Multiple Regression Analysis

Author: Luka Herceg | Business Analytics, University of Amsterdam
Tools: Python 3 - pandas, numpy, scikit-learn, matplotlib, seaborn
Dataset: Ames Housing Dataset - 2,930 real residential property sales (Ames, Iowa, 2006-2010)

---

## Project Overview

This project builds and evaluates a house price prediction model using the Ames Housing Dataset - a widely-used real-world benchmark in regression and machine learning.

Three regression models are estimated and compared:
- OLS (Ordinary Least Squares) - baseline multiple linear regression
- Ridge Regression (L2) - regularized model to handle potential overfitting
- Lasso Regression (L1) - regularized model with automatic feature selection

---

## Results

| Model | R2 (Test Set) | RMSE |
|-------|--------------|------|
| OLS Regression | 0.8616 | $33,309 |
| Ridge (alpha=50) | 0.8623 | $33,229 |
| Lasso (alpha=100) | 0.8620 | $33,262 |

5-fold Cross-Validation R2: 0.8397 - confirms the model generalises well.

---

## Key Findings

- Overall Quality is the single strongest predictor of sale price (r = 0.80)
- Above-ground living area is the second most important feature (r = 0.71)
- Year Built has a strong positive effect - newer properties command higher prices
- Neighbourhood is highly significant - top areas command $100K+ premiums
- All three models perform nearly identically, confirming minimal overfitting

---

## Files

| File | Description |
|------|-------------|
| ames_house_price_analysis.py | Full Python analysis script |
| Ames_House_Price_Prediction_Report.pdf | Professional report with visualisations and findings |

---

## How to Run

Install dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn

Run the analysis:
python ames_house_price_analysis.py

The script expects AmesHousing.csv in the same directory.
Download the dataset from: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset

---

## Methodology

1. Data Cleaning - median imputation for missing numeric values, rows with missing categoricals dropped
2. Feature Engineering - one-hot encoding of categorical variables (Neighborhood, House Style, Central Air, Kitchen Quality)
3. Standardization - StandardScaler fitted on training set only (no data leakage)
4. Train/Test Split - 80% train (2,344 obs.) / 20% test (586 obs.), random seed = 42
5. Model Estimation - OLS, Ridge, Lasso via scikit-learn
6. Evaluation - R2, RMSE, MAE, 5-fold cross-validation

---

This project is part of my freelance data analytics portfolio.
Available for hire on Upwork: https://www.upwork.com/freelancers/~0188e797f303106ffb
