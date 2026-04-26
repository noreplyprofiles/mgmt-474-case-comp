"""
v7 — key changes:
  • CatBoost gets raw Surname/Geography/Gender as native cat_features
    → uses CatBoost's ordered TS (more principled than manual OOF encoding)
    → also tries grow_policy='Lossguide' (leaf-wise, like LGB)
  • Two CatBoost variants: Depthwise (default) + Lossguide
  • Removed DART LGB and meta-learner (both hurt in v6)
  • 4-seed ensemble: XGB + LGB + CAT_depth + CAT_loss
  • Simple OOF-weighted average, raw probability averaging
  • Raw OOF target encoding for XGB/LGB only
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

BASE_SEED = 42
N_FOLDS   = 5
SEEDS     = [42, 123, 456, 789]

# ── Load data ─────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
y     = train['Exited'].astype(int)
n_train     = len(train)
global_mean = y.mean()

# ── Raw OOF target encoding (for XGB / LGB only) ──────────────────────────────
def target_encode_oof(s_train, y_train, s_test):
    gm  = y_train.mean()
    cv  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
    oof = np.full(len(s_train), gm)
    for tr, val in cv.split(s_train, y_train):
        m = y_train.iloc[tr].groupby(s_train.iloc[tr].values).mean().to_dict()
        oof[val] = s_train.iloc[val].map(m).fillna(gm).values
    full_map = y_train.groupby(s_train.values).mean().to_dict()
    return oof, s_test.map(full_map).fillna(gm).values

surname_oof,  surname_test  = target_encode_oof(train['Surname'],   y, test['Surname'])
geo_oof,      geo_test      = target_encode_oof(train['Geography'], y, test['Geography'])
gender_oof,   gender_test   = target_encode_oof(train['Gender'],    y, test['Gender'])

# ── Quantile-rank features ────────────────────────────────────────────────────
all_data     = pd.concat([train.drop(columns=['Exited']), test], ignore_index=True)
surname_freq = pd.concat([train['Surname'], test['Surname']]).value_counts().to_dict()
RANK_COLS    = ['Age', 'Balance', 'CreditScore', 'EstimatedSalary', 'Tenure']
rank_all     = {c: rankdata(all_data[c].values) / len(all_data) for c in RANK_COLS}

# ── Feature engineering (shared base) ────────────────────────────────────────
def engineer_base(df, offset):
    """Returns engineered DataFrame WITHOUT target-encoded categoricals."""
    df = df.copy(); n = len(df)
    # Numeric encodings of cats (for XGB/LGB use)
    df['Gender_bin']  = (df['Gender'] == 'Male').astype(int)
    df['Geo_France']  = (df['Geography'] == 'France').astype(int)
    df['Geo_Germany'] = (df['Geography'] == 'Germany').astype(int)
    df['Geo_Spain']   = (df['Geography'] == 'Spain').astype(int)
    df['SurnameFreq'] = df['Surname'].map(surname_freq).fillna(1)
    for c in RANK_COLS:
        df[f'{c}_qrank'] = rank_all[c][offset: offset + n]
    df['Prod1']        = (df['NumOfProducts'] == 1).astype(int)
    df['Prod2']        = (df['NumOfProducts'] == 2).astype(int)
    df['Prod3or4']     = (df['NumOfProducts'] >= 3).astype(int)
    df['ProdHighRisk'] = ((df['NumOfProducts'] == 1) | (df['NumOfProducts'] >= 3)).astype(int)
    df['HasBalance']      = (df['Balance'] > 0).astype(int)
    df['HighBalance']     = (df['Balance'] > 100_000).astype(int)
    df['VeryHighBalance'] = (df['Balance'] > 150_000).astype(int)
    df['LogBalance']      = np.log1p(df['Balance'])
    df['SeniorCustomer'] = (df['Age'] > 50).astype(int)
    df['MidAge']         = ((df['Age'] >= 35) & (df['Age'] <= 50)).astype(int)
    df['Age2']           = df['Age'] ** 2
    df['AgeGroup']       = pd.cut(df['Age'],
        bins=[0,25,30,35,40,45,50,55,60,100], labels=False).fillna(0)
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['CreditScorePerAge']  = df['CreditScore'] / (df['Age'] + 1)
    df['ProductsPerTenure']  = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['TenureAgeRatio']     = df['Tenure'] / (df['Age'] + 1)
    df['BalancePerProduct']  = df['Balance'] / (df['NumOfProducts'] + 1)
    df['SalaryPerAge']       = df['EstimatedSalary'] / (df['Age'] + 1)
    df['LogCreditScore']     = np.log1p(df['CreditScore'])
    df['LogSalary']          = np.log1p(df['EstimatedSalary'])
    df['IsGermanyInactive']  = (df['Geo_Germany'] * (1 - df['IsActiveMember'])).astype(int)
    df['GermanyBalance']     = df['Geo_Germany'] * df['Balance']
    df['GermanyAge']         = df['Geo_Germany'] * df['Age']
    df['GermanyProd3or4']    = df['Geo_Germany'] * df['Prod3or4']
    df['AgeBalance']         = df['Age'] * df['Balance']
    df['AgeInactive']        = df['Age'] * (1 - df['IsActiveMember'])
    df['BalanceInactive']    = df['Balance'] * (1 - df['IsActiveMember'])
    df['SeniorInactive']     = df['SeniorCustomer'] * (1 - df['IsActiveMember'])
    df['SeniorHighBalance']  = df['SeniorCustomer'] * df['HighBalance']
    df['SeniorProd3or4']     = df['SeniorCustomer'] * df['Prod3or4']
    df['AgeProd3or4']        = df['Age'] * df['Prod3or4']
    df['MultiProduct']       = (df['NumOfProducts'] > 1).astype(int)
    df['InactiveBalance']    = (1 - df['IsActiveMember']) * df['HasBalance']
    df['TenureProduct']      = df['Tenure'] * df['NumOfProducts']
    df['AgeSalary']          = df['Age'] * df['EstimatedSalary']
    df['AgeProd1']           = df['Age'] * df['Prod1']
    df['BalanceProd']        = df['Balance'] * df['NumOfProducts']
    drop_cols = ['id', 'CustomerId', 'Geography', 'Gender']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

# XGB / LGB features: base + manual target encoding
def make_gbm_feats(df, sur_enc, geo_enc, gen_enc, offset):
    base = engineer_base(df, offset)
    # drop Surname (will use target-encoded version)
    base.drop(columns=['Surname'], errors='ignore', inplace=True)
    base['SurnameTargetEnc'] = sur_enc
    base['GeoTargetEnc']     = geo_enc
    base['GenderTargetEnc']  = gen_enc
    # rank-interaction features
    base['AgeRank_x_GeoEnc']  = base['Age_qrank'] * geo_enc
    base['BalRank_x_Inactive'] = base['Balance_qrank'] * (1 - base['IsActiveMember'])
    base['AgeRank_x_Prod3or4'] = base['Age_qrank'] * base['Prod3or4']
    base['SurnameEnc_x_Prod']  = sur_enc * base['NumOfProducts']
    base['GeoEnc_x_Age']       = geo_enc * base['Age']
    base['GeoEnc_x_Balance']   = geo_enc * base['Balance']
    return base.apply(pd.to_numeric, errors='coerce').fillna(0)

# CatBoost features: base + RAW categorical columns (let CatBoost encode them)
def make_cat_feats(df, sur_enc, geo_enc, offset):
    base = engineer_base(df, offset)
    # keep raw Surname for CatBoost's native TS — leave as string
    # add rank interactions using the manual geo encoding for cross-features
    base['AgeRank_x_GeoEnc']  = base['Age_qrank'] * geo_enc
    base['BalRank_x_Inactive'] = base['Balance_qrank'] * (1 - base['IsActiveMember'])
    base['AgeRank_x_Prod3or4'] = base['Age_qrank'] * base['Prod3or4']
    base['SurnameFreq_x_Prod'] = base['SurnameFreq'] * base['NumOfProducts']
    # Also include manual OOF as extra feature
    base['SurnameTargetEnc']   = sur_enc
    base['GeoTargetEnc']       = geo_enc
    # numeric convert all except Surname
    surname_col = base['Surname'].copy()
    base.drop(columns=['Surname'], inplace=True)
    base = base.apply(pd.to_numeric, errors='coerce').fillna(0)
    base['Surname'] = surname_col.values
    return base

# Build feature matrices
X_gbm      = make_gbm_feats(train.drop(columns=['Exited']), surname_oof,  geo_oof,  gender_oof,  0)
X_gbm_test = make_gbm_feats(test,                            surname_test, geo_test, gender_test, n_train)
X_cat_df   = make_cat_feats(train.drop(columns=['Exited']), surname_oof,  geo_oof,  0)
X_cat_test_df = make_cat_feats(test,                         surname_test, geo_test, n_train)

X_gbm_arr      = X_gbm.values.astype(float)
X_gbm_test_arr = X_gbm_test.values.astype(float)

# CatBoost needs Pool with cat_features
surname_col_idx = list(X_cat_df.columns).index('Surname')
CAT_FEATURE_IDX = [surname_col_idx]
print(f"GBM features: {X_gbm.shape[1]}  |  CatBoost features: {X_cat_df.shape[1]}")
print(f"CatBoost categorical column index: {CAT_FEATURE_IDX} → 'Surname'")
print(f"Train shape: {X_gbm.shape} | Churn rate: {y.mean():.1%}")

# ── Optuna HPO ────────────────────────────────────────────────────────────────
PREV_LGB = dict(n_estimators=2612, learning_rate=0.003169, max_depth=4, num_leaves=81,
                subsample=0.6206, colsample_bytree=0.461, min_child_samples=54,
                reg_alpha=0.3435, reg_lambda=2.042)
PREV_XGB = dict(n_estimators=937, learning_rate=0.01345, max_depth=3,
                subsample=0.6001, colsample_bytree=0.9861, min_child_weight=3,
                gamma=0.1426, reg_alpha=0.00222, reg_lambda=0.3425)

def inner_auc_gbm(model_cls, params):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=BASE_SEED)
    return np.mean([roc_auc_score(y.iloc[v],
        model_cls(**params).fit(X_gbm_arr[t], y.iloc[t]).predict_proba(X_gbm_arr[v])[:,1])
        for t, v in inner.split(X_gbm_arr, y)])

def inner_auc_cat(params):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=BASE_SEED)
    scores = []
    for t, v in inner.split(X_cat_df, y):
        pool_tr = cb.Pool(X_cat_df.iloc[t], y.iloc[t], cat_features=CAT_FEATURE_IDX)
        pool_vl = cb.Pool(X_cat_df.iloc[v],             cat_features=CAT_FEATURE_IDX)
        m = cb.CatBoostClassifier(**params)
        m.fit(pool_tr)
        scores.append(roc_auc_score(y.iloc[v], m.predict_proba(pool_vl)[:,1]))
    return np.mean(scores)

print("\nOptimizing LightGBM (50 trials)...")
def lgb_obj(trial):
    return inner_auc_gbm(lgb.LGBMClassifier, dict(
        n_estimators=trial.suggest_int('n_estimators',500,4000),
        learning_rate=trial.suggest_float('learning_rate',0.003,0.05,log=True),
        max_depth=trial.suggest_int('max_depth',3,8),
        num_leaves=trial.suggest_int('num_leaves',20,120),
        subsample=trial.suggest_float('subsample',0.5,1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree',0.4,1.0),
        min_child_samples=trial.suggest_int('min_child_samples',5,60),
        reg_alpha=trial.suggest_float('reg_alpha',1e-3,10.0,log=True),
        reg_lambda=trial.suggest_float('reg_lambda',1e-3,10.0,log=True),
        random_state=BASE_SEED, verbose=-1, n_jobs=-1))
sl = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
sl.enqueue_trial(PREV_LGB); sl.optimize(lgb_obj, n_trials=50, show_progress_bar=False)
best_lgb = sl.best_params
print(f"  Best LGB: {sl.best_value:.4f}")

print("\nOptimizing XGBoost (50 trials)...")
def xgb_obj(trial):
    return inner_auc_gbm(xgb.XGBClassifier, dict(
        n_estimators=trial.suggest_int('n_estimators',500,4000),
        learning_rate=trial.suggest_float('learning_rate',0.003,0.05,log=True),
        max_depth=trial.suggest_int('max_depth',3,7),
        subsample=trial.suggest_float('subsample',0.5,1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree',0.4,1.0),
        min_child_weight=trial.suggest_int('min_child_weight',1,15),
        gamma=trial.suggest_float('gamma',0.0,0.5),
        reg_alpha=trial.suggest_float('reg_alpha',1e-3,10.0,log=True),
        reg_lambda=trial.suggest_float('reg_lambda',1e-3,10.0,log=True),
        eval_metric='auc', random_state=BASE_SEED, tree_method='hist', n_jobs=-1))
sx = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
sx.enqueue_trial(PREV_XGB); sx.optimize(xgb_obj, n_trials=50, show_progress_bar=False)
best_xgb = sx.best_params
print(f"  Best XGB: {sx.best_value:.4f}")

print("\nOptimizing CatBoost Depthwise (50 trials, native cats)...")
def cat_depth_obj(trial):
    return inner_auc_cat(dict(
        iterations=trial.suggest_int('iterations',500,3000),
        learning_rate=trial.suggest_float('learning_rate',0.003,0.08,log=True),
        depth=trial.suggest_int('depth',4,8),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg',0.5,15.0),
        subsample=trial.suggest_float('subsample',0.6,1.0),
        colsample_bylevel=trial.suggest_float('colsample_bylevel',0.5,1.0),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf',5,30),
        grow_policy='SymmetricTree',
        random_seed=BASE_SEED, verbose=0, eval_metric='AUC'))
scd = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
scd.optimize(cat_depth_obj, n_trials=50, show_progress_bar=False)
best_cat_d = scd.best_params
print(f"  Best CAT Depthwise: {scd.best_value:.4f}")

print("\nOptimizing CatBoost Lossguide (50 trials, native cats)...")
def cat_loss_obj(trial):
    return inner_auc_cat(dict(
        iterations=trial.suggest_int('iterations',500,3000),
        learning_rate=trial.suggest_float('learning_rate',0.003,0.08,log=True),
        max_leaves=trial.suggest_int('max_leaves',16,128),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg',0.5,15.0),
        subsample=trial.suggest_float('subsample',0.6,1.0),
        colsample_bylevel=trial.suggest_float('colsample_bylevel',0.5,1.0),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf',5,30),
        grow_policy='Lossguide',
        random_seed=BASE_SEED, verbose=0, eval_metric='AUC'))
scl = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
scl.optimize(cat_loss_obj, n_trials=50, show_progress_bar=False)
best_cat_l = scl.best_params
print(f"  Best CAT Lossguide: {scl.best_value:.4f}")

# ── Model factory ─────────────────────────────────────────────────────────────
def make_gbm(seed):
    return {
        'xgb': xgb.XGBClassifier(**best_xgb, eval_metric='auc',
                   random_state=seed, tree_method='hist', n_jobs=-1),
        'lgb': lgb.LGBMClassifier(**best_lgb, random_state=seed, verbose=-1, n_jobs=-1),
    }

def make_cat(seed):
    return {
        'cat_d': cb.CatBoostClassifier(**best_cat_d, random_seed=seed, verbose=0),
        'cat_l': cb.CatBoostClassifier(**best_cat_l, random_seed=seed, verbose=0),
    }

GBM_NAMES = ['xgb', 'lgb']
CAT_NAMES = ['cat_d', 'cat_l']
ALL_NAMES = GBM_NAMES + CAT_NAMES

oof_acc  = {nm: np.zeros(n_train)   for nm in ALL_NAMES}
test_acc = {nm: np.zeros(len(test)) for nm in ALL_NAMES}

X_cat_arr      = X_cat_df.values
X_cat_test_arr = X_cat_test_df.values

print(f"\n── Multi-seed OOF ({len(SEEDS)} seeds × {N_FOLDS} folds × {len(ALL_NAMES)} models) ──")
for seed in SEEDS:
    gbm_models = make_gbm(seed)
    cat_models = make_cat(seed)
    cv         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof   = {nm: np.zeros(n_train)   for nm in ALL_NAMES}
    seed_tst   = {nm: np.zeros(len(test)) for nm in ALL_NAMES}
    print(f"\n  seed={seed}")

    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_gbm_arr, y), 1):
        yf_tr  = y.iloc[tr_idx].values
        yf_val = y.iloc[val_idx].values
        print(f"    fold {fold_i}", end="")

        # GBM models
        for nm, model in gbm_models.items():
            m = clone(model)
            m.fit(X_gbm_arr[tr_idx], yf_tr)
            seed_oof[nm][val_idx] = m.predict_proba(X_gbm_arr[val_idx])[:, 1]
            seed_tst[nm]         += m.predict_proba(X_gbm_test_arr)[:, 1] / N_FOLDS
            print(f"  {nm}={roc_auc_score(yf_val, seed_oof[nm][val_idx]):.4f}", end="")

        # CatBoost models with native cat features
        for nm, model in cat_models.items():
            pool_tr = cb.Pool(X_cat_arr[tr_idx],  yf_tr,  cat_features=CAT_FEATURE_IDX)
            pool_vl = cb.Pool(X_cat_arr[val_idx],          cat_features=CAT_FEATURE_IDX)
            pool_te = cb.Pool(X_cat_test_arr,               cat_features=CAT_FEATURE_IDX)
            m = clone(model)
            m.fit(pool_tr)
            seed_oof[nm][val_idx] = m.predict_proba(pool_vl)[:, 1]
            seed_tst[nm]         += m.predict_proba(pool_te)[:, 1] / N_FOLDS
            print(f"  {nm}={roc_auc_score(yf_val, seed_oof[nm][val_idx]):.4f}", end="")
        print()

    seed_aucs = {nm: roc_auc_score(y, seed_oof[nm]) for nm in ALL_NAMES}
    print(f"  seed OOF: {' '.join(f'{nm}={v:.4f}' for nm,v in seed_aucs.items())}")
    for nm in ALL_NAMES:
        oof_acc[nm]  += seed_oof[nm]  / len(SEEDS)
        test_acc[nm] += seed_tst[nm]  / len(SEEDS)

print("\n── Final OOF AUC (avg across seeds) ─────────────────────")
oof_aucs = {nm: roc_auc_score(y, oof_acc[nm]) for nm in ALL_NAMES}
for nm, auc in oof_aucs.items():
    print(f"  {nm:10s}: {auc:.4f}")

# Weighted average by OOF AUC
w          = np.array([oof_aucs[nm] for nm in ALL_NAMES]); w /= w.sum()
oof_stack  = np.column_stack([oof_acc[nm]  for nm in ALL_NAMES])
test_stack = np.column_stack([test_acc[nm] for nm in ALL_NAMES])
final_oof  = oof_stack  @ w
final_test = test_stack @ w
print(f"\nWeighted ensemble OOF AUC: {roc_auc_score(y, final_oof):.4f}")
print(f"Weights: {dict(zip(ALL_NAMES, w.round(3)))}")

# ── Save submission ────────────────────────────────────────────────────────────
sub = pd.read_csv('sample_submission.csv')
sub['Exited'] = final_test
sub.to_csv('submission.csv', index=False)
print(f"\nDone! submission.csv saved.")
print(sub.head())
