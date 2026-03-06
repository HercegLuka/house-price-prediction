"""
House Price Prediction: Multiple Regression Analysis
Dataset: Ames Housing Dataset (2,930 real property sales, Ames, Iowa)
Author: Luka Herceg | Business Analytics, University of Amsterdam
Tools: Python — pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD & CLEAN DATA ────────────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/AmesHousing.csv')

# Select key features
features_num = ['Gr Liv Area', 'Total Bsmt SF', 'Garage Area', 'Lot Area',
                'Year Built', 'Year Remod/Add', 'Overall Qual', 'Overall Cond',
                'Full Bath', 'Bedroom AbvGr', 'TotRms AbvGrd', 'Fireplaces']
features_cat = ['Neighborhood', 'House Style', 'Central Air', 'Kitchen Qual']
target = 'SalePrice'

df_model = df[features_num + features_cat + [target]].copy()

# Fill missing numeric values with median
for col in features_num:
    df_model[col] = df_model[col].fillna(df_model[col].median())

# Drop rows with missing categorical
df_model = df_model.dropna(subset=features_cat)
df_model = df_model[df_model[target] > 0].reset_index(drop=True)

# Encode categoricals
df_encoded = pd.get_dummies(df_model, columns=features_cat, drop_first=True)
df_encoded.columns = [c.replace(' ', '_') for c in df_encoded.columns]

# Features & target
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# ─── SPLIT & SCALE ────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── MODELS ───────────────────────────────────────────────────────────────────
ols   = LinearRegression().fit(X_train_sc, y_train)
ridge = Ridge(alpha=50).fit(X_train_sc, y_train)
lasso = Lasso(alpha=100, max_iter=10000).fit(X_train_sc, y_train)

y_pred_ols   = ols.predict(X_test_sc)
y_pred_ridge = ridge.predict(X_test_sc)
y_pred_lasso = lasso.predict(X_test_sc)

r2_ols    = r2_score(y_test, y_pred_ols)
rmse_ols  = np.sqrt(mean_squared_error(y_test, y_pred_ols))
mae_ols   = mean_absolute_error(y_test, y_pred_ols)
r2_ridge  = r2_score(y_test, y_pred_ridge)
rmse_ridge= np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_lasso  = r2_score(y_test, y_pred_lasso)
rmse_lasso= np.sqrt(mean_squared_error(y_test, y_pred_lasso))
cv_r2     = cross_val_score(ols, X_train_sc, y_train, cv=5, scoring='r2').mean()

residuals = y_test - y_pred_ols

# Top feature importances (OLS standardized coefs, numeric features only)
num_feature_names = features_num
coef_df = pd.Series(
    ols.coef_[:len(num_feature_names)],
    index=[f.replace(' ', '_') for f in num_feature_names]
).abs().sort_values(ascending=True).tail(10)

# ─── STYLE ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.facecolor': '#FAFAFA',
})
BLUE   = '#1F5C99'
ORANGE = '#E87722'
GREEN  = '#2E8B57'
RED    = '#CC3333'
LIGHT  = '#EEF4FB'

# ══════════════════════════════════════════════════════════════
# FIGURE 1 — EDA (2x2)
# ══════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.patch.set_facecolor('#FAFAFA')
fig1.suptitle('Ames Housing Dataset — Exploratory Data Analysis\n2,930 Real Property Sales | Ames, Iowa',
              fontsize=15, fontweight='bold', color='#1a1a2e', y=0.99)

# 1a — Price distribution
ax = axes[0, 0]
ax.hist(df_model['SalePrice'] / 1000, bins=50, color=BLUE, alpha=0.85, edgecolor='white', linewidth=0.4)
ax.axvline(df_model['SalePrice'].median() / 1000, color=ORANGE, lw=2.5, linestyle='--',
           label=f"Median: ${df_model['SalePrice'].median()/1000:.0f}K")
ax.axvline(df_model['SalePrice'].mean() / 1000, color=RED, lw=2, linestyle=':',
           label=f"Mean: ${df_model['SalePrice'].mean()/1000:.0f}K")
ax.set_xlabel('Sale Price (USD thousands)', fontsize=11)
ax.set_ylabel('Number of Properties', fontsize=11)
ax.set_title('Distribution of Sale Prices', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

# 1b — Price vs Living Area (coloured by Overall Quality)
ax = axes[0, 1]
sc = ax.scatter(df_model['Gr Liv Area'], df_model['SalePrice'] / 1000,
                c=df_model['Overall Qual'], cmap='Blues', alpha=0.5, s=12)
plt.colorbar(sc, ax=ax, label='Overall Quality (1–10)')
ax.set_xlabel('Above-Ground Living Area (sq ft)', fontsize=11)
ax.set_ylabel('Sale Price (USD thousands)', fontsize=11)
ax.set_title('Sale Price vs. Living Area\n(coloured by Overall Quality)', fontsize=13, fontweight='bold')

# 1c — Avg price by top 10 neighbourhoods
ax = axes[1, 0]
top_neigh = (df_model.groupby('Neighborhood')['SalePrice']
             .mean().sort_values(ascending=False).head(10) / 1000)
bars = ax.barh(top_neigh.index, top_neigh.values, color=BLUE, alpha=0.85, edgecolor='white')
ax.set_xlabel('Avg Sale Price (USD thousands)', fontsize=11)
ax.set_title('Top 10 Neighbourhoods by Avg Sale Price', fontsize=13, fontweight='bold')
for bar, val in zip(bars, top_neigh.values):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2,
            f'${val:.0f}K', va='center', fontsize=8)

# 1d — Correlation heatmap (numeric features vs SalePrice)
ax = axes[1, 1]
corr_cols = ['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF',
             'Garage Area', 'Year Built', 'Full Bath', 'TotRms AbvGrd']
corr = df_model[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            mask=mask, linewidths=0.5, annot_kws={'size': 8})
ax.set_title('Correlation Matrix — Key Variables', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=35, labelsize=8)
ax.tick_params(axis='y', rotation=0, labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/home/claude/ames_fig1_eda.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()

# ══════════════════════════════════════════════════════════════
# FIGURE 2 — MODEL RESULTS (2x2)
# ══════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.patch.set_facecolor('#FAFAFA')
fig2.suptitle('Ames Housing — Regression Model Results\nOLS | Ridge | Lasso',
              fontsize=15, fontweight='bold', color='#1a1a2e', y=0.99)

# 2a — Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test / 1000, y_pred_ols / 1000, alpha=0.4, s=15, color=BLUE, label='OLS predictions')
lim = [30, 800]
ax.plot(lim, lim, 'r--', lw=2, label='Perfect prediction')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Actual Price ($K)', fontsize=11)
ax.set_ylabel('Predicted Price ($K)', fontsize=11)
ax.set_title(f'Actual vs. Predicted Prices\nOLS  R² = {r2_ols:.4f}', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

# 2b — Residuals
ax = axes[0, 1]
ax.hist(residuals / 1000, bins=45, color=ORANGE, alpha=0.85, edgecolor='white', linewidth=0.4)
ax.axvline(0, color='red', lw=2, linestyle='--')
ax.set_xlabel('Residual (USD thousands)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Residual Distribution\n(Should be centred at zero)', fontsize=13, fontweight='bold')
ax.text(0.97, 0.95,
        f'Mean: ${residuals.mean()/1000:.1f}K\nStd: ${residuals.std()/1000:.1f}K\nSkew: {pd.Series(residuals).skew():.2f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor=LIGHT, alpha=0.9))

# 2c — Feature importance (|standardized coef|, numeric only)
ax = axes[1, 0]
colors_bar = [BLUE if v > 0 else RED for v in ols.coef_[:len(num_feature_names)]]
coef_signed = pd.Series(
    ols.coef_[:len(num_feature_names)],
    index=[f.replace(' ', '_') for f in num_feature_names]
).sort_values()
bar_colors = [BLUE if v > 0 else RED for v in coef_signed.values]
coef_signed.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='white', alpha=0.85)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Standardized Coefficient', fontsize=11)
ax.set_title('Feature Coefficients\n(Blue = positive effect, Red = negative)', fontsize=13, fontweight='bold')
ax.tick_params(axis='y', labelsize=9)

# 2d — Model comparison (R² and RMSE)
ax = axes[1, 1]
model_names = ['OLS', 'Ridge\n(α=50)', 'Lasso\n(α=100)']
r2s   = [r2_ols,   r2_ridge,   r2_lasso]
rmses = [rmse_ols, rmse_ridge, rmse_lasso]
x = np.arange(3)
ax2 = ax.twinx()
b1 = ax.bar(x - 0.2, r2s,        0.35, color=BLUE,   alpha=0.85, label='R²')
b2 = ax2.bar(x + 0.2, [r/1000 for r in rmses], 0.35, color=ORANGE, alpha=0.85, label='RMSE ($K)')
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel('R² Score', color=BLUE, fontsize=11)
ax2.set_ylabel('RMSE (USD thousands)', color=ORANGE, fontsize=11)
ax.set_ylim(0.7, 1.0)
ax.set_title('Model Comparison: OLS vs Ridge vs Lasso', fontsize=13, fontweight='bold')
ax.grid(False); ax2.grid(False)
for bar, val in zip(b1, r2s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
for bar, val in zip(b2, [r/1000 for r in rmses]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'${val:.1f}K', ha='center', fontsize=9, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc='lower right', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/home/claude/ames_fig2_model.png', dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()

print(f"""
=== AMES HOUSING — MODEL SUMMARY ===
Dataset:   {len(df_model)} real property sales (Ames, Iowa)
Features:  {X.shape[1]} (after encoding)
Split:     80% train / 20% test

OLS Regression:
  R²:      {r2_ols:.4f}
  RMSE:    ${rmse_ols:,.0f}
  MAE:     ${mae_ols:,.0f}
  CV R²:   {cv_r2:.4f}

Ridge (α=50):
  R²:      {r2_ridge:.4f}
  RMSE:    ${rmse_ridge:,.0f}

Lasso (α=100):
  R²:      {r2_lasso:.4f}
  RMSE:    ${rmse_lasso:,.0f}

Top numeric predictors: {coef_df.tail(3).index.tolist()}
""")
