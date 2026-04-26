"""
v8 — lean model: fewer features, better generalization
  • REMOVE surname target encoding (main source of OOF-LB gap)
  • Keep geography target encoding (only 3 values, very stable)
  • Only keep features with clear predictive logic (~35 features)
  • More aggressive regularization
  • 4 seeds × 3 models (XGB + LGB + CAT)
  • Hypothesis: OOF drops slightly but LB gap shrinks more → net win
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
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

# ── Geography target encoding only (3 values — very stable, minimal overfitting)
def target_encode_oof(s_train, y_train, s_test):
    gm  = y_train.mean()
    cv  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
    oof = np.full(len(s_train), gm)
    for tr, val in cv.split(s_train, y_train):
        m = y_train.iloc[tr].groupby(s_train.iloc[tr].values).mean().to_dict()
        oof[val] = s_train.iloc[val].map(m).fillna(gm).values
    full_map = y_train.groupby(s_train.values).mean().to_dict()
    return oof, s_test.map(full_map).fillna(gm).values

geo_oof,  geo_test  = target_encode_oof(train['Geography'], y, test['Geography'])
# gender is only 2 values — also stable
gender_oof, gender_test = target_encode_oof(train['Gender'], y, test['Gender'])

# ── Quantile-rank features ────────────────────────────────────────────────────
all_data     = pd.concat([train.drop(columns=['Exited']), test], ignore_index=True)
surname_freq = pd.concat([train['Surname'], test['Surname']]).value_counts().to_dict()
RANK_COLS    = ['Age', 'Balance', 'CreditScore', 'EstimatedSalary']
rank_all     = {c: rankdata(all_data[c].values) / len(all_data) for c in RANK_COLS}

# ── Lean feature set ──────────────────────────────────────────────────────────
def engineer(df, geo_enc, gen_enc, offset):
    df = df.copy(); n = len(df)

    # Stable target encodings (low cardinality)
    df['GeoTargetEnc']    = geo_enc
    df['GenderTargetEnc'] = gen_enc

    # Surname: frequency only (no target encoding = no leakage risk)
    df['SurnameFreq'] = df['Surname'].map(surname_freq).fillna(1)
    df['RareSurname'] = (df['SurnameFreq'] == 1).astype(int)

    # Geography binary flags
    df['Geo_Germany'] = (df['Geography'] == 'Germany').astype(int)
    df['Geo_France']  = (df['Geography'] == 'France').astype(int)
    df['Gender_bin']  = (df['Gender'] == 'Male').astype(int)

    # Quantile ranks
    for c in RANK_COLS:
        df[f'{c}_qrank'] = rank_all[c][offset: offset + n]

    # Product flags (massive signal)
    df['Prod1']        = (df['NumOfProducts'] == 1).astype(int)
    df['Prod2']        = (df['NumOfProducts'] == 2).astype(int)
    df['Prod3or4']     = (df['NumOfProducts'] >= 3).astype(int)
    df['ProdHighRisk'] = ((df['NumOfProducts'] == 1) | (df['NumOfProducts'] >= 3)).astype(int)

    # Balance flags
    df['HasBalance']   = (df['Balance'] > 0).astype(int)
    df['HighBalance']  = (df['Balance'] > 100_000).astype(int)
    df['LogBalance']   = np.log1p(df['Balance'])

    # Age flags
    df['SeniorCustomer'] = (df['Age'] > 50).astype(int)
    df['MidAge']         = ((df['Age'] >= 35) & (df['Age'] <= 50)).astype(int)
    df['YoungCustomer']  = (df['Age'] < 30).astype(int)

    # Core ratios
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['CreditScorePerAge']  = df['CreditScore'] / (df['Age'] + 1)
    df['BalancePerProduct']  = df['Balance'] / (df['NumOfProducts'] + 1)
    df['TenureAgeRatio']     = df['Tenure'] / (df['Age'] + 1)

    # Key proven interactions
    df['IsGermanyInactive'] = (df['Geo_Germany'] * (1 - df['IsActiveMember'])).astype(int)
    df['GermanyAge']        = df['Geo_Germany'] * df['Age']
    df['GermanyBalance']    = df['Geo_Germany'] * df['Balance']
    df['GermanyProd3or4']   = df['Geo_Germany'] * df['Prod3or4']
    df['AgeInactive']       = df['Age'] * (1 - df['IsActiveMember'])
    df['BalanceInactive']   = df['Balance'] * (1 - df['IsActiveMember'])
    df['SeniorInactive']    = df['SeniorCustomer'] * (1 - df['IsActiveMember'])
    df['SeniorProd3or4']    = df['SeniorCustomer'] * df['Prod3or4']
    df['AgeProd3or4']       = df['Age'] * df['Prod3or4']
    df['SeniorHighBalance'] = df['SeniorCustomer'] * df['HighBalance']
    df['AgeBalance']        = df['Age'] * df['Balance']
    df['BalanceProd']       = df['Balance'] * df['NumOfProducts']
    df['GeoEnc_x_Age']      = geo_enc * df['Age']
    df['AgeRank_x_Prod3or4']= df['Age_qrank'] * df['Prod3or4']
    df['BalRank_x_Inactive']= df['Balance_qrank'] * (1 - df['IsActiveMember'])

    drop_cols = ['id', 'CustomerId', 'Surname', 'Geography', 'Gender']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df.apply(pd.to_numeric, errors='coerce').fillna(0)

X      = engineer(train.drop(columns=['Exited']), geo_oof,  gender_oof,  0)
X_test = engineer(test,                            geo_test, gender_test, n_train)
X_arr      = X.values.astype(float)
X_test_arr = X_test.values.astype(float)
print(f"Train shape: {X.shape} | Churn rate: {y.mean():.1%}")
print(f"Features: {list(X.columns)}")

# ── Optuna HPO (more regularization bias) ────────────────────────────────────
def inner_auc(model_cls, params):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=BASE_SEED)
    return np.mean([roc_auc_score(y.iloc[v],
        model_cls(**params).fit(X_arr[t], y.iloc[t]).predict_proba(X_arr[v])[:,1])
        for t, v in inner.split(X_arr, y)])

print("\nOptimizing LightGBM (60 trials)...")
def lgb_obj(trial):
    return inner_auc(lgb.LGBMClassifier, dict(
        n_estimators=trial.suggest_int('n_estimators',300,3000),
        learning_rate=trial.suggest_float('learning_rate',0.003,0.05,log=True),
        max_depth=trial.suggest_int('max_depth',3,7),
        num_leaves=trial.suggest_int('num_leaves',15,80),
        subsample=trial.suggest_float('subsample',0.5,0.9),
        colsample_bytree=trial.suggest_float('colsample_bytree',0.4,0.9),
        min_child_samples=trial.suggest_int('min_child_samples',10,80),
        reg_alpha=trial.suggest_float('reg_alpha',1e-2,10.0,log=True),
        reg_lambda=trial.suggest_float('reg_lambda',1e-2,10.0,log=True),
        random_state=BASE_SEED, verbose=-1, n_jobs=-1))
sl = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
sl.optimize(lgb_obj, n_trials=60, show_progress_bar=False)
best_lgb = sl.best_params
print(f"  Best LGB: {sl.best_value:.4f}")

print("\nOptimizing XGBoost (60 trials)...")
def xgb_obj(trial):
    return inner_auc(xgb.XGBClassifier, dict(
        n_estimators=trial.suggest_int('n_estimators',300,3000),
        learning_rate=trial.suggest_float('learning_rate',0.003,0.05,log=True),
        max_depth=trial.suggest_int('max_depth',3,6),
        subsample=trial.suggest_float('subsample',0.5,0.9),
        colsample_bytree=trial.suggest_float('colsample_bytree',0.4,0.9),
        min_child_weight=trial.suggest_int('min_child_weight',3,20),
        gamma=trial.suggest_float('gamma',0.0,1.0),
        reg_alpha=trial.suggest_float('reg_alpha',1e-2,10.0,log=True),
        reg_lambda=trial.suggest_float('reg_lambda',1e-2,10.0,log=True),
        eval_metric='auc', random_state=BASE_SEED, tree_method='hist', n_jobs=-1))
sx = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
sx.optimize(xgb_obj, n_trials=60, show_progress_bar=False)
best_xgb = sx.best_params
print(f"  Best XGB: {sx.best_value:.4f}")

print("\nOptimizing CatBoost (60 trials)...")
def cat_obj(trial):
    return inner_auc(cb.CatBoostClassifier, dict(
        iterations=trial.suggest_int('iterations',300,2500),
        learning_rate=trial.suggest_float('learning_rate',0.003,0.08,log=True),
        depth=trial.suggest_int('depth',4,7),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1.0,20.0),
        subsample=trial.suggest_float('subsample',0.6,0.95),
        colsample_bylevel=trial.suggest_float('colsample_bylevel',0.5,0.95),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf',5,40),
        random_seed=BASE_SEED, verbose=0, eval_metric='AUC'))
sc = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
sc.optimize(cat_obj, n_trials=60, show_progress_bar=False)
best_cat = sc.best_params
print(f"  Best CAT: {sc.best_value:.4f}")

# ── Model factory ─────────────────────────────────────────────────────────────
def make_models(seed):
    return {
        'xgb': xgb.XGBClassifier(**best_xgb, eval_metric='auc',
                   random_state=seed, tree_method='hist', n_jobs=-1),
        'lgb': lgb.LGBMClassifier(**best_lgb, random_state=seed, verbose=-1, n_jobs=-1),
        'cat': cb.CatBoostClassifier(**best_cat, eval_metric='AUC',
                   random_seed=seed, verbose=0),
    }

MODEL_NAMES = list(make_models(BASE_SEED).keys())

# ── Multi-seed OOF ────────────────────────────────────────────────────────────
oof_acc  = {nm: np.zeros(n_train)   for nm in MODEL_NAMES}
test_acc = {nm: np.zeros(len(test)) for nm in MODEL_NAMES}

print(f"\n── Multi-seed OOF ({len(SEEDS)} seeds × {N_FOLDS} folds × {len(MODEL_NAMES)} models) ──")
for seed in SEEDS:
    models   = make_models(seed)
    cv       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = {nm: np.zeros(n_train)   for nm in MODEL_NAMES}
    seed_tst = {nm: np.zeros(len(test)) for nm in MODEL_NAMES}
    print(f"\n  seed={seed}")
    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_arr, y), 1):
        yf_tr  = y.iloc[tr_idx].values
        yf_val = y.iloc[val_idx].values
        print(f"    fold {fold_i}", end="")
        for nm, model in models.items():
            m = clone(model)
            m.fit(X_arr[tr_idx], yf_tr)
            seed_oof[nm][val_idx] = m.predict_proba(X_arr[val_idx])[:, 1]
            seed_tst[nm]         += m.predict_proba(X_test_arr)[:, 1] / N_FOLDS
            print(f"  {nm}={roc_auc_score(yf_val, seed_oof[nm][val_idx]):.4f}", end="")
        print()
    seed_aucs = {nm: roc_auc_score(y, seed_oof[nm]) for nm in MODEL_NAMES}
    print(f"  seed OOF: {' '.join(f'{nm}={v:.4f}' for nm,v in seed_aucs.items())}")
    for nm in MODEL_NAMES:
        oof_acc[nm]  += seed_oof[nm]  / len(SEEDS)
        test_acc[nm] += seed_tst[nm]  / len(SEEDS)

print("\n── Final OOF AUC ────────────────────────────────────────")
oof_aucs = {nm: roc_auc_score(y, oof_acc[nm]) for nm in MODEL_NAMES}
for nm, auc in oof_aucs.items():
    print(f"  {nm:5s}: {auc:.4f}")

w          = np.array([oof_aucs[nm] for nm in MODEL_NAMES]); w /= w.sum()
final_test = np.column_stack([test_acc[nm] for nm in MODEL_NAMES]) @ w
final_oof  = np.column_stack([oof_acc[nm]  for nm in MODEL_NAMES]) @ w
print(f"\nWeighted ensemble OOF AUC: {roc_auc_score(y, final_oof):.4f}")
print(f"Weights: {dict(zip(MODEL_NAMES, w.round(3)))}")

# ── Save submission ────────────────────────────────────────────────────────────
sub = pd.read_csv('sample_submission.csv')
sub['Exited'] = final_test
sub.to_csv('submission.csv', index=False)
print(f"\nDone! submission.csv saved.")
print(sub.head())
