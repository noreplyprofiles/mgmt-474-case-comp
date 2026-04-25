import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

SEED     = 42
N_FOLDS  = 5

# ── Load data ─────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
y     = train['Exited'].astype(int)
n_train     = len(train)
global_mean = y.mean()

# ── Target encoding helper (OOF to avoid leakage) ─────────────────────────────
def target_encode_oof(series_train, y_train, series_test, n_folds=5, seed=42):
    """Returns (oof_encoded_train, encoded_test)."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    gm  = y_train.mean()
    oof = np.full(len(series_train), gm)
    for tr_idx, val_idx in cv.split(series_train, y_train):
        fold_map = (y_train.iloc[tr_idx]
                    .groupby(series_train.iloc[tr_idx].values)
                    .mean().to_dict())
        oof[val_idx] = series_train.iloc[val_idx].map(fold_map).fillna(gm).values
    full_map = y_train.groupby(series_train.values).mean().to_dict()
    test_enc = series_test.map(full_map).fillna(gm).values
    return oof, test_enc

# Encode categorical columns via OOF target encoding
surname_oof,  surname_test  = target_encode_oof(
    train['Surname'],   y, test['Surname'])
geo_oof,      geo_test      = target_encode_oof(
    train['Geography'], y, test['Geography'])
gender_oof,   gender_test   = target_encode_oof(
    train['Gender'],    y, test['Gender'])

# ── Feature engineering ───────────────────────────────────────────────────────
all_surnames     = pd.concat([train['Surname'], test['Surname']])
surname_freq_map = all_surnames.value_counts().to_dict()

def engineer(df, surname_target, geo_target, gender_target):
    df = df.copy()

    # Basic encodings
    df['Gender_bin']  = (df['Gender'] == 'Male').astype(int)
    df['Geo_France']  = (df['Geography'] == 'France').astype(int)
    df['Geo_Germany'] = (df['Geography'] == 'Germany').astype(int)
    df['Geo_Spain']   = (df['Geography'] == 'Spain').astype(int)

    # OOF target encodings
    df['SurnameTargetEnc'] = surname_target
    df['GeoTargetEnc']     = geo_target
    df['GenderTargetEnc']  = gender_target
    df['SurnameFreq']      = df['Surname'].map(surname_freq_map).fillna(1)

    # Product features — NumOfProducts 1 & 3/4 = high churn, 2 = low churn
    df['Prod1']        = (df['NumOfProducts'] == 1).astype(int)
    df['Prod2']        = (df['NumOfProducts'] == 2).astype(int)
    df['Prod3or4']     = (df['NumOfProducts'] >= 3).astype(int)
    df['ProdHighRisk'] = ((df['NumOfProducts'] == 1) | (df['NumOfProducts'] >= 3)).astype(int)

    # Balance features
    df['HasBalance']      = (df['Balance'] > 0).astype(int)
    df['HighBalance']     = (df['Balance'] > 100_000).astype(int)
    df['VeryHighBalance'] = (df['Balance'] > 150_000).astype(int)
    df['ZeroBalance']     = (df['Balance'] == 0).astype(int)
    df['LogBalance']      = np.log1p(df['Balance'])

    # Age features
    df['SeniorCustomer'] = (df['Age'] > 50).astype(int)
    df['MidAge']         = ((df['Age'] >= 35) & (df['Age'] <= 50)).astype(int)
    df['Age2']           = df['Age'] ** 2
    df['AgeGroup']       = pd.cut(
        df['Age'],
        bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 100],
        labels=False
    ).fillna(0)

    # Ratio features
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['CreditScorePerAge']  = df['CreditScore'] / (df['Age'] + 1)
    df['ProductsPerTenure']  = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['TenureAgeRatio']     = df['Tenure'] / (df['Age'] + 1)
    df['BalancePerProduct']  = df['Balance'] / (df['NumOfProducts'] + 1)
    df['SalaryPerAge']       = df['EstimatedSalary'] / (df['Age'] + 1)
    df['LogCreditScore']     = np.log1p(df['CreditScore'])
    df['LogSalary']          = np.log1p(df['EstimatedSalary'])

    # Interaction features
    df['IsGermanyInactive'] = (df['Geo_Germany'] * (1 - df['IsActiveMember'])).astype(int)
    df['GermanyBalance']    = df['Geo_Germany'] * df['Balance']
    df['GermanyAge']        = df['Geo_Germany'] * df['Age']
    df['GermanyProd3or4']   = df['Geo_Germany'] * df['Prod3or4']
    df['AgeBalance']        = df['Age'] * df['Balance']
    df['AgeInactive']       = df['Age'] * (1 - df['IsActiveMember'])
    df['BalanceInactive']   = df['Balance'] * (1 - df['IsActiveMember'])
    df['SeniorInactive']    = df['SeniorCustomer'] * (1 - df['IsActiveMember'])
    df['SeniorHighBalance'] = df['SeniorCustomer'] * df['HighBalance']
    df['SeniorProd3or4']    = df['SeniorCustomer'] * df['Prod3or4']
    df['AgeProd3or4']       = df['Age'] * df['Prod3or4']
    df['MultiProduct']      = (df['NumOfProducts'] > 1).astype(int)
    df['InactiveBalance']   = (1 - df['IsActiveMember']) * df['HasBalance']
    df['TenureProduct']     = df['Tenure'] * df['NumOfProducts']
    df['AgeSalary']         = df['Age'] * df['EstimatedSalary']
    df['CreditAge2']        = df['CreditScore'] * df['Age2']
    df['GeoEnc_x_Age']      = geo_target * df['Age']
    df['GeoEnc_x_Balance']  = geo_target * df['Balance']
    df['SurnameEnc_x_Prod'] = surname_target * df['NumOfProducts']
    df['Tenure_is_0']       = (df['Tenure'] == 0).astype(int)
    df['Tenure_is_10']      = (df['Tenure'] == 10).astype(int)
    df['AgeProd1']          = df['Age'] * df['Prod1']
    df['BalanceProd']       = df['Balance'] * df['NumOfProducts']
    df['CreditBalanceProd'] = df['CreditScore'] * df['HasBalance'] * df['NumOfProducts']

    drop_cols = ['id', 'CustomerId', 'Surname', 'Geography', 'Gender']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

X      = engineer(train.drop(columns=['Exited']), surname_oof,  geo_oof,  gender_oof)
X_test = engineer(test,                            surname_test, geo_test, gender_test)

X      = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"Train shape: {X.shape} | Churn rate: {y.mean():.1%}")

# ── Optuna hyperparameter search for LightGBM ────────────────────────────────
print("\nOptimizing LightGBM hyperparameters with Optuna (50 trials)...")
X_arr = X.values.astype(float)

def lgb_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int('n_estimators', 500, 3000),
        learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        max_depth         = trial.suggest_int('max_depth', 4, 8),
        num_leaves        = trial.suggest_int('num_leaves', 20, 100),
        subsample         = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_samples = trial.suggest_int('min_child_samples', 10, 50),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        random_state=SEED, verbose=-1, n_jobs=-1,
    )
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr, val in cv_inner.split(X_arr, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_arr[tr], y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[val], m.predict_proba(X_arr[val])[:, 1]))
    return np.mean(scores)

study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgb.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
best_lgb  = study_lgb.best_params
print(f"  Best LGB OOF (3-fold): {study_lgb.best_value:.4f}  params={best_lgb}")

# ── Optuna hyperparameter search for XGBoost ─────────────────────────────────
print("\nOptimizing XGBoost hyperparameters with Optuna (50 trials)...")

def xgb_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int('n_estimators', 500, 3000),
        learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        max_depth         = trial.suggest_int('max_depth', 3, 7),
        subsample         = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 10),
        gamma             = trial.suggest_float('gamma', 0.0, 0.5),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        eval_metric='auc', random_state=SEED, tree_method='hist', n_jobs=-1,
    )
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr, val in cv_inner.split(X_arr, y):
        m = xgb.XGBClassifier(**params)
        m.fit(X_arr[tr], y.iloc[tr])
        scores.append(roc_auc_score(y.iloc[val], m.predict_proba(X_arr[val])[:, 1]))
    return np.mean(scores)

study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(xgb_objective, n_trials=50, show_progress_bar=False)
best_xgb  = study_xgb.best_params
print(f"  Best XGB OOF (3-fold): {study_xgb.best_value:.4f}  params={best_xgb}")

# ── Final model definitions (tuned + CatBoost) ────────────────────────────────
models = {
    'xgb': xgb.XGBClassifier(
        **best_xgb,
        eval_metric='auc', random_state=SEED, tree_method='hist', n_jobs=-1,
    ),
    'lgb': lgb.LGBMClassifier(
        **best_lgb,
        random_state=SEED, verbose=-1, n_jobs=-1,
    ),
    'cat': cb.CatBoostClassifier(
        iterations=2000, learning_rate=0.02,
        depth=6, l2_leaf_reg=3.0,
        subsample=0.8, colsample_bylevel=0.7,
        min_data_in_leaf=15, eval_metric='AUC',
        random_seed=SEED, verbose=0,
    ),
}

# ── OOF training & test predictions ──────────────────────────────────────────
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
X_test_arr = X_test.values.astype(float)

oof_preds  = {name: np.zeros(n_train)   for name in models}
test_preds = {name: np.zeros(len(test)) for name in models}

for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_arr, y), 1):
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y.iloc[tr_idx].values, y.iloc[val_idx].values
    print(f"\n── Fold {fold_i} ─────────────────────────")

    for name, model in models.items():
        m = clone(model)
        m.fit(X_tr, y_tr)
        oof_preds[name][val_idx] = m.predict_proba(X_val)[:, 1]
        test_preds[name] += m.predict_proba(X_test_arr)[:, 1] / N_FOLDS
        fold_auc = roc_auc_score(y_val, oof_preds[name][val_idx])
        print(f"  {name:5s}: {fold_auc:.4f}")

print("\n── OOF AUC ──────────────────────────────")
oof_aucs = {}
for name in models:
    oof_aucs[name] = roc_auc_score(y, oof_preds[name])
    print(f"  {name:5s}: {oof_aucs[name]:.4f}")

# ── Stacking ensemble ─────────────────────────────────────────────────────────
oof_stack  = np.column_stack([oof_preds[n] for n in models])
test_stack = np.column_stack([test_preds[n] for n in models])

w = np.array([oof_aucs[n] for n in models])
w = w / w.sum()
weighted_oof  = oof_stack @ w
weighted_test = test_stack @ w
w_auc = roc_auc_score(y, weighted_oof)
print(f"\nWeighted avg OOF AUC : {w_auc:.4f}  weights={dict(zip(models, w.round(3)))}")

meta = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
meta.fit(oof_stack, y)
meta_oof  = meta.predict_proba(oof_stack)[:, 1]
meta_test = meta.predict_proba(test_stack)[:, 1]
m_auc = roc_auc_score(y, meta_oof)
print(f"Meta-learner OOF AUC : {m_auc:.4f}")

if m_auc >= w_auc:
    final = meta_test
    print("→ Using meta-learner")
else:
    final = weighted_test
    print("→ Using weighted average")

# ── Save submission ────────────────────────────────────────────────────────────
sub = pd.read_csv('sample_submission.csv')
sub['Exited'] = final
sub.to_csv('submission.csv', index=False)
print(f"\nDone! submission.csv saved.")
print(sub.head())
