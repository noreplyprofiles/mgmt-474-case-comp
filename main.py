import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# ── Feature engineering ──────────────────────────────────────────────────────
def feature_engineering(df):
    df = df.copy()

    # Encode categoricals
    df['Gender']        = (df['Gender'] == 'Male').astype(int)
    df['Geo_France']    = (df['Geography'] == 'France').astype(int)
    df['Geo_Germany']   = (df['Geography'] == 'Germany').astype(int)
    df['Geo_Spain']     = (df['Geography'] == 'Spain').astype(int)

    # New features
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['CreditScorePerAge']  = df['CreditScore'] / (df['Age'] + 1)
    df['ProductsPerTenure']  = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['HasBalance']         = (df['Balance'] > 0).astype(int)
    df['AgeGroup']           = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=False)
    df['IsGermanyInactive']  = ((df['Geo_Germany'] == 1) & (df['IsActiveMember'] == 0)).astype(int)
    df['MultiProduct']       = (df['NumOfProducts'] > 1).astype(int)

    # Drop columns that aren't useful for prediction
    drop_cols = ['id', 'CustomerId', 'Surname', 'Geography']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df

X      = feature_engineering(train.drop(columns=['Exited']))
y      = train['Exited'].astype(int)
X_test = feature_engineering(test)

print(f"Train shape: {X.shape} | Churn rate: {y.mean():.1%}")

# ── Models ───────────────────────────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='auc',
    random_state=42
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)

# ── Cross-validation ─────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc')
lgb_scores  = cross_val_score(lgb_model, X, y, cv=cv, scoring='roc_auc')

print(f"XGBoost  CV AUC: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
print(f"LightGBM CV AUC: {lgb_scores.mean():.4f} ± {lgb_scores.std():.4f}")

# ── Train on full data & ensemble ─────────────────────────────────────────────
xgb_model.fit(X, y)
lgb_model.fit(X, y)

xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
lgb_preds  = lgb_model.predict_proba(X_test)[:, 1]
ensemble   = 0.5 * xgb_preds + 0.5 * lgb_preds

# ── Save submission ───────────────────────────────────────────────────────────
sub = pd.read_csv('sample_submission.csv')
sub['Exited'] = ensemble
sub.to_csv('submission.csv', index=False)
print(f"\nDone! submission.csv saved.")
print(sub.head())