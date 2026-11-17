"""
Environmental ML Project: Predicting Beijing PM2.5 from weather
Plain-Python version of notebooks/ML_analysis.ipynb (same code, step by step).
Run from the src/ folder:  python ML_analysis.py
"""


# ======================================================================
# Predicting Beijing PM2.5 Air Pollution from Weather with Machine Learning
# ======================================================================

# ======================================================================
# Step 1: Import Libraries
# ======================================================================
# (Google Colab) uncomment if a library is missing
# !pip install -q xgboost lightgbm shap

import os, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# scikit-learn: preprocessing, models, tuning, metrics
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# Gradient boosting libraries + explainable ML
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap

RANDOM_STATE = 42
FIG_DIR = '../figures/' if os.path.isdir('../figures') else 'figures/'
os.makedirs(FIG_DIR, exist_ok=True)

# colour-blind-safe palette + clean chart style
BLUE, ORANGE, AQUA, YELLOW, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a8985'
DIVERGING = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#eb6834'])
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'axes.titleweight': 'bold'})

def save(name):
    plt.savefig(FIG_DIR + name)
    plt.show()


# ======================================================================
# Step 2: Import the Dataset
# ======================================================================
DATA_URL = 'https://raw.githubusercontent.com/6730467221/Environmental-ML-Project/main/data/beijing_pm25.csv'
local_path = '../data/beijing_pm25.csv'

df = pd.read_csv(local_path) if os.path.exists(local_path) else pd.read_csv(DATA_URL)
print(df.head())


# ======================================================================
# Step 3: Explore the Dataset
# ======================================================================
# Check dataset size
print(df.shape)

# Check data types and missing values
df.info()

# Summary statistics
print(df.describe().round(1).T)

# Missing values
print(df.isna().sum())
print(f"\nPM2.5 missing: {df['pm2.5'].isna().mean():.1%}")

# Build a datetime index
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
eda = df.set_index('datetime')

# Daily mean PM2.5 over 5 years
daily = eda['pm2.5'].resample('D').mean()
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(daily.index, daily, color=BLUE, lw=0.8)
ax.axhline(37.5, color=ORANGE, lw=1.5, ls='--')
ax.text(daily.index[10], 50, 'Thai 24-h standard 37.5 µg/m³', color='#52514e', fontsize=9)
ax.set_title('Daily mean PM2.5 in Beijing, 2010–2014'); ax.set_ylabel('PM2.5 (µg/m³)')
save('01_pm25_timeseries.png')
print(f"Days above Thai standard: {(daily > 37.5).mean():.0%}")

# Seasonal (monthly) and daily (hourly) pattern
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
m = eda.groupby('month')['pm2.5'].mean()
axes[0].bar(m.index, m.values, color=[ORANGE if v > 110 else BLUE for v in m.values], width=0.7)
axes[0].set_xticks(range(1, 13)); axes[0].set_xlabel('Month'); axes[0].set_ylabel('Mean PM2.5 (µg/m³)')
axes[0].set_title('Seasonal pattern: winter is worst')
h = eda.groupby('hour')['pm2.5'].mean()
axes[1].plot(h.index, h.values, color=BLUE, lw=2, marker='o', ms=4)
axes[1].set_xticks(range(0, 24, 3)); axes[1].set_xlabel('Hour of day'); axes[1].set_ylabel('Mean PM2.5 (µg/m³)')
axes[1].set_title('Daily cycle: night-time build-up')
plt.tight_layout()
save('02_seasonal_daily_pattern.png')

# Distribution + wind direction
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].hist(eda['pm2.5'].dropna(), bins=80, color=BLUE, edgecolor='white', linewidth=0.3)
axes[0].set_xlabel('PM2.5 (µg/m³)'); axes[0].set_ylabel('Hours'); axes[0].set_title('PM2.5 distribution (right-skewed)')
sns.boxplot(data=eda, x='cbwd', y='pm2.5', order=['NW', 'NE', 'SE', 'cv'], ax=axes[1], showfliers=False,
            palette=[BLUE, AQUA, ORANGE, YELLOW], width=0.55)
axes[1].set_xlabel('Wind direction (cv = calm/variable)'); axes[1].set_ylabel('PM2.5 (µg/m³)')
axes[1].set_title('NW wind cleans the air')
plt.tight_layout()
save('03_distribution_wind.png')
print(eda.groupby('cbwd')['pm2.5'].mean().round(0).sort_values())

# Correlation matrix
num_cols = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(eda[num_cols].corr(), annot=True, fmt='.2f', cmap=DIVERGING, vmin=-1, vmax=1, ax=ax,
            linewidths=2, linecolor='white')
ax.set_title('Correlation matrix')
save('04_correlation.png')


# ======================================================================
# Step 4: Data Preprocessing
# ======================================================================
data = df.drop(columns=['No']).set_index('datetime').sort_index()
print('From', data.index.min(), 'to', data.index.max(), '| rows:', len(data))


# ======================================================================
# Step 5: Feature Engineering
# ======================================================================
# Relative humidity from temperature and dew point (Magnus formula)
a, b = 17.625, 243.04
data['RH'] = (100 * np.exp(a * data['DEWP'] / (b + data['DEWP']))
                  / np.exp(a * data['TEMP'] / (b + data['TEMP']))).clip(0, 100)
data['dew_spread'] = data['TEMP'] - data['DEWP']

# Rolling (past) weather: 24 h and 72 h means
for col in ['DEWP', 'Iws', 'PRES', 'TEMP', 'RH']:
    data[f'{col}_24h'] = data[col].rolling(24, min_periods=1).mean()
for col in ['DEWP', 'Iws']:
    data[f'{col}_72h'] = data[col].rolling(72, min_periods=1).mean()

data['PRES_chg24'] = data['PRES'].diff(24)                                   # pressure change (fronts)
data['rain_24h'] = (data['Ir'].rolling(24, min_periods=1).max() > 0).astype(int)

# Hours since the last north-westerly (clean) wind = how long air has been stagnant
is_nw = (data['cbwd'] == 'NW').astype(int)
data['hrs_since_NW'] = is_nw.groupby(is_nw.cumsum()).cumcount().where(is_nw == 0, 0).clip(upper=240)

# Cyclic time features + one-hot wind direction
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data['dayofweek'] = data.index.dayofweek
data = pd.get_dummies(data, columns=['cbwd'], prefix='wind', dtype=int)

# Remove rows without a target
data = data.dropna(subset=['pm2.5', 'PRES_chg24'])
print(data.shape)

BASE_FEATURES = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'wind_NE', 'wind_NW', 'wind_SE', 'wind_cv',
                 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dayofweek']
ENGINEERED = ['RH', 'dew_spread', 'DEWP_24h', 'DEWP_72h', 'Iws_24h', 'Iws_72h', 'PRES_24h', 'PRES_chg24',
              'TEMP_24h', 'RH_24h', 'hrs_since_NW', 'rain_24h']
FEATURES = BASE_FEATURES + ENGINEERED

X = data[FEATURES]
y = data['pm2.5']
print('Features:', len(FEATURES))


# ======================================================================
# Step 6: Train/Test Split (time-based)
# ======================================================================
train_mask = data['year'] < 2014
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]
print("Training:", X_train.shape)
print("Testing:", X_test.shape)

# Helper: evaluate a model on the test set and store results
results = {}
predictions = {}

def evaluate(name, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
    predictions[name] = y_pred
    print(name)
    print("R²:", round(r2, 3))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))


# ======================================================================
# Step 7: Linear Regression (baseline)
# ======================================================================
lr_model = Pipeline([
    ('scale', StandardScaler()),
    ('model', LinearRegression())
])
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

evaluate('Linear Regression', y_pred_lr)


# ======================================================================
# Step 8: K-Nearest Neighbours (KNN)
# ======================================================================
knn_model = Pipeline([
    ('scale', StandardScaler()),
    ('model', KNeighborsRegressor(n_neighbors=15, n_jobs=-1))
])
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

evaluate('KNN', y_pred_knn)


# ======================================================================
# Step 9: Decision Tree Regression
# ======================================================================
dt_model = DecisionTreeRegressor(
    max_depth=10,
    random_state=RANDOM_STATE
)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

evaluate('Decision Tree', y_pred_dt)


# ======================================================================
# Step 10: Random Forest Regression
# ======================================================================
rf_model = RandomForestRegressor(
    n_estimators=200,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

evaluate('Random Forest', y_pred_rf)


# ======================================================================
# Step 11: Histogram Gradient Boosting
# ======================================================================
hgb_model = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE
)
hgb_model.fit(X_train, y_train)

y_pred_hgb = hgb_model.predict(X_test)

evaluate('Hist Gradient Boosting', y_pred_hgb)


# ======================================================================
# Step 12: XGBoost Regression
# ======================================================================
xgb_model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

evaluate('XGBoost', y_pred_xgb)


# ======================================================================
# Step 13: LightGBM Regression
# ======================================================================
lgbm_model = LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    verbose=-1
)
lgbm_model.fit(X_train, y_train)

y_pred_lgbm = lgbm_model.predict(X_test)

evaluate('LightGBM', y_pred_lgbm)


# ======================================================================
# Step 14: Compare the 7 Models
# ======================================================================
baseline_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
print(baseline_df.round(3))


# ======================================================================
# Step 15: Effect of Feature Engineering
# ======================================================================
fe_model = LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)
fe_model.fit(X_train[BASE_FEATURES], y_train)
r2_raw = r2_score(y_test, fe_model.predict(X_test[BASE_FEATURES]))
r2_eng = results['LightGBM']['R2']
print(f"Raw features only : R² = {r2_raw:.3f}")
print(f"Raw + engineered  : R² = {r2_eng:.3f}")

fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(['Raw features only', 'Raw + engineered'], [r2_raw, r2_eng], color=[GREY, BLUE], height=0.5)
for i, v in enumerate([r2_raw, r2_eng]):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center')
ax.set_xlim(0, 0.75); ax.set_xlabel('Test R²'); ax.set_title('Effect of feature engineering (LightGBM)')
ax.grid(axis='y', visible=False)
save('05_feature_engineering_effect.png')


# ======================================================================
# Step 16: Hyperparameter Tuning
# ======================================================================
tscv = TimeSeriesSplit(n_splits=3)


# ======================================================================
# 16.1 Tune Random Forest
# ======================================================================
rf_params = {
    'n_estimators': [150, 250],
    'max_depth': [12, 18, 25, None],
    'min_samples_leaf': [1, 3, 5, 10],
    'max_features': ['sqrt', 0.33, 0.5],
}
rf_search = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE),
                               rf_params, n_iter=8, cv=tscv, scoring='r2', random_state=RANDOM_STATE)
rf_search.fit(X_train, y_train)
print("Best params:", rf_search.best_params_)
print("CV R²:", round(rf_search.best_score_, 3))
evaluate('Random Forest (tuned)', rf_search.best_estimator_.predict(X_test))


# ======================================================================
# 16.2 Tune XGBoost
# ======================================================================
xgb_params = {
    'n_estimators': [300, 500, 800],
    'learning_rate': [0.02, 0.03, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 5, 10],
}
xgb_search = RandomizedSearchCV(XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                                xgb_params, n_iter=15, cv=tscv, scoring='r2', random_state=RANDOM_STATE)
xgb_search.fit(X_train, y_train)
print("Best params:", xgb_search.best_params_)
print("CV R²:", round(xgb_search.best_score_, 3))
evaluate('XGBoost (tuned)', xgb_search.best_estimator_.predict(X_test))


# ======================================================================
# 16.3 Tune LightGBM
# ======================================================================
lgbm_params = {
    'n_estimators': [300, 500, 800],
    'learning_rate': [0.02, 0.03, 0.05, 0.1],
    'num_leaves': [15, 31, 63],
    'min_child_samples': [10, 20, 50],
    'subsample': [0.7, 0.85, 1.0], 'subsample_freq': [1],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [0, 1, 5],
}
lgbm_search = RandomizedSearchCV(LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
                                 lgbm_params, n_iter=15, cv=tscv, scoring='r2', random_state=RANDOM_STATE)
lgbm_search.fit(X_train, y_train)
print("Best params:", lgbm_search.best_params_)
print("CV R²:", round(lgbm_search.best_score_, 3))
evaluate('LightGBM (tuned)', lgbm_search.best_estimator_.predict(X_test))


# ======================================================================
# Step 17: Model Evaluation (all models)
# ======================================================================
all_results = pd.DataFrame(results).T.sort_values('R2', ascending=False)
all_results.to_csv(FIG_DIR + 'model_results.csv')
print(all_results.round(3))

r = all_results.sort_values('R2')
best_name = r.index[-1]
colors = [ORANGE if n == best_name else (AQUA if 'tuned' in n else BLUE) for n in r.index]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].barh(r.index, r['R2'], color=colors, height=0.6)
for i, v in enumerate(r['R2']):
    axes[0].text(v + 0.008, i, f'{v:.2f}', va='center', fontsize=9)
axes[0].set_xlim(0, r['R2'].max() + 0.12); axes[0].set_xlabel('Test R² (higher is better)')
axes[0].set_title('Model comparison — R²'); axes[0].grid(axis='y', visible=False)
axes[1].barh(r.index, r['RMSE'], color=colors, height=0.6)
for i, v in enumerate(r['RMSE']):
    axes[1].text(v + 0.8, i, f'{v:.1f}', va='center', fontsize=9)
axes[1].set_xlabel('Test RMSE, µg/m³ (lower is better)'); axes[1].set_title('Model comparison — RMSE')
axes[1].set_yticklabels([]); axes[1].grid(axis='y', visible=False)
plt.tight_layout()
save('06_model_comparison.png')
print("Best model:", best_name)

best_model = lgbm_search.best_estimator_
y_pred = predictions[best_name]

fig, ax = plt.subplots(figsize=(5.5, 5.2))
ax.scatter(y_test, y_pred, s=6, alpha=0.25, color=BLUE, edgecolors='none')
ax.plot([0, 650], [0, 650], color=ORANGE, lw=1.5, ls='--', label='Perfect prediction')
ax.set_xlim(0, 650); ax.set_ylim(0, 650)
ax.set_xlabel('Observed PM2.5 (µg/m³)'); ax.set_ylabel('Predicted PM2.5 (µg/m³)')
ax.set_title(f'{best_name}: observed vs predicted (2014)'); ax.legend(frameon=False)
save('07_observed_vs_predicted.png')

pred_series = pd.Series(y_pred, index=y_test.index)
obs, pred = y_test.loc['2014-01-01':'2014-03-31'], pred_series.loc['2014-01-01':'2014-03-31']
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.plot(obs.index, obs, color=GREY, lw=1, label='Observed')
ax.plot(pred.index, pred, color=BLUE, lw=1.4, label='Predicted')
ax.set_ylabel('PM2.5 (µg/m³)'); ax.set_title('Winter 2014 (Jan–Mar): the model tracks haze episodes from weather alone')
ax.legend(frameon=False, ncol=2, loc='upper right')
save('08_timeseries_prediction.png')

# Early warning: can the model flag unhealthy hours (PM2.5 > 75 µg/m³)?
obs_bad, pred_bad = y_test > 75, y_pred > 75
print(f"Accuracy : {(obs_bad == pred_bad).mean():.1%}")
print(f"Recall   : {(obs_bad & pred_bad).sum() / obs_bad.sum():.1%}")
print(f"Precision: {(obs_bad & pred_bad).sum() / pred_bad.sum():.1%}")


# ======================================================================
# Step 18: Feature Importance (Explainable ML)
# ======================================================================

# ======================================================================
# 18.1 Permutation importance
# ======================================================================
perm = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, scoring='r2')
perm_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values().tail(15)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(perm_imp.index, perm_imp.values, color=BLUE, height=0.6)
ax.set_xlabel('Drop in test R² when shuffled'); ax.set_title('Permutation importance (top 15)')
ax.grid(axis='y', visible=False)
save('09_permutation_importance.png')


# ======================================================================
# 18.2 SHAP values
# ======================================================================
X_shap = X_test.sample(2000, random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap)

plt.figure()
shap.summary_plot(shap_values, X_shap, max_display=15, show=False, plot_size=(8, 6))
plt.title('SHAP summary: how each feature moves PM2.5 predictions')
save('10_shap_summary.png')

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(mean_abs_shap.index, mean_abs_shap.values, color=BLUE, height=0.6)
ax.set_xlabel('Mean |SHAP value| (µg/m³)'); ax.set_title('Top 10 drivers of PM2.5 (SHAP)')
ax.grid(axis='y', visible=False)
save('11_shap_top10.png')
print(mean_abs_shap.sort_values(ascending=False).round(1))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, feat, label, title in zip(
        axes, ['hrs_since_NW', 'RH', 'Iws_24h'],
        ['Hours since last NW wind', 'Relative humidity (%)', '24-h mean cumulated wind speed (m/s)'],
        ['Stagnation builds pollution', 'Humid air → more PM2.5', 'Wind disperses pollution']):
    i = FEATURES.index(feat)
    ax.scatter(X_shap[feat], shap_values[:, i], s=8, alpha=0.4, color=BLUE, edgecolors='none')
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel(label); ax.set_ylabel('SHAP value (µg/m³)'); ax.set_title(title)
axes[2].set_xscale('log')
plt.tight_layout()
save('12_shap_dependence.png')


# ======================================================================
# Step 19: Environmental Interpretation
# ======================================================================