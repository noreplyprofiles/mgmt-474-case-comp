"""
v6 — key changes from v5:
  • DART-mode LightGBM (dropouts reduce overfitting, adds diversity)
  • Optuna-tuned MLP (50 trials — was untuned before)
  • LGB stacking meta-learner: OOF preds + 8 key raw features → LGB
  • Reduced target encoding noise: surname alpha=10 (light smoothing only)
  • Same 4 SEEDS × 5 folds for base models
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

# ── Lightly smoothed OOF target encoding (alpha=10 reduces rare-surname noise) ─
def smooth_te_oof(s_train, y_train, s_test, alpha=10):
    gm  = y_train.mean()
    cv  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
    oof = np.full(len(s_train), gm)
    for tr, val in cv.split(s_train, y_train):
        stats = (pd.DataFrame({'v': s_train.iloc[tr].values,
                               't': y_train.iloc[tr].values})
                   .groupby('v')['t'].agg(['count','mean']))
        smap  = ((stats['count'] * stats['mean'] + alpha * gm) /
                 (stats['count'] + alpha)).to_dict()
        oof[val] = s_train.iloc[val].map(smap).fillna(gm).values
    stats = (pd.DataFrame({'v': s_train.values, 't': y_train.values})
               .groupby('v')['t'].agg(['count','mean']))
    fmap = ((stats['count'] * stats['mean'] + alpha * gm) /
            (stats['count'] + alpha)).to_dict()
    return oof, s_test.map(fmap).fillna(gm).values

surname_oof,  surname_test  = smooth_te_oof(train['Surname'],   y, test['Surname'],   alpha=10)
geo_oof,      geo_test      = smooth_te_oof(train['Geography'], y, test['Geography'], alpha=2)
gender_oof,   gender_test   = smooth_te_oof(train['Gender'],    y, test['Gender'],    alpha=2)

# ── Quantile-rank features (combined train+test, no leakage) ─────────────────
all_data     = pd.concat([train.drop(columns=['Exited']), test], ignore_index=True)
surname_freq = pd.concat([train['Surname'], test['Surname']]).value_counts().to_dict()
RANK_COLS    = ['Age', 'Balance', 'CreditScore', 'EstimatedSalary', 'Tenure']
rank_all     = {c: rankdata(all_data[c].values) / len(all_data) for c in RANK_COLS}

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer(df, sur_enc, geo_enc, gen_enc, offset):
    df = df.copy(); n = len(df)
    df['Gender_bin']  = (df['Gender'] == 'Male').astype(int)
    df['Geo_France']  = (df['Geography'] == 'France').astype(int)
    df['Geo_Germany'] = (df['Geography'] == 'Germany').astype(int)
    df['Geo_Spain']   = (df['Geography'] == 'Spain').astype(int)
    df['SurnameTargetEnc'] = sur_enc
    df['GeoTargetEnc']     = geo_enc
    df['GenderTargetEnc']  = gen_enc
    df['SurnameFreq']      = df['Surname'].map(surname_freq).fillna(1)
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
    df['CreditAge2']         = df['CreditScore'] * df['Age2']
    df['GeoEnc_x_Age']       = geo_enc * df['Age']
    df['GeoEnc_x_Balance']   = geo_enc * df['Balance']
    df['SurnameEnc_x_Prod']  = sur_enc * df['NumOfProducts']
    df['Tenure_is_0']        = (df['Tenure'] == 0).astype(int)
    df['AgeProd1']           = df['Age'] * df['Prod1']
    df['BalanceProd']        = df['Balance'] * df['NumOfProducts']
    df['CreditBalanceProd']  = df['CreditScore'] * df['HasBalance'] * df['NumOfProducts']
    df['AgeRank_x_GeoEnc']  = df['Age_qrank'] * geo_enc
    df['BalRank_x_Inactive'] = df['Balance_qrank'] * (1 - df['IsActiveMember'])
    df['AgeRank_x_Prod3or4'] = df['Age_qrank'] * df['Prod3or4']
    drop_cols = ['id', 'CustomerId', 'Surname', 'Geography', 'Gender']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

X      = engineer(train.drop(columns=['Exited']), surname_oof,  geo_oof,  gender_oof,  0)
X_test = engineer(test,                            surname_test, geo_test, gender_test, n_train)
X      = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
X_arr      = X.values.astype(float)
X_test_arr = X_test.values.astype(float)
scaler        = StandardScaler()
X_scaled      = scaler.fit_transform(X_arr)
X_test_scaled = scaler.transform(X_test_arr)
print(f"Train shape: {X.shape} | Churn rate: {y.mean():.1%}")

# ── Optuna HPO ────────────────────────────────────────────────────────────────
PREV_LGB = dict(n_estimators=2612, learning_rate=0.003169, max_depth=4, num_leaves=81,
                subsample=0.6206, colsample_bytree=0.461, min_child_samples=54,
                reg_alpha=0.3435, reg_lambda=2.042)
PREV_XGB = dict(n_estimators=937,  learning_rate=0.01345, max_depth=3,
                subsample=0.6001, colsample_bytree=0.9861, min_child_weight=3,
                gamma=0.1426, reg_alpha=0.00222, reg_lambda=0.3425)

def inner_auc(model_cls, params, X_in=None):
    if X_in is None: X_in = X_arr
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=BASE_SEED)
    return np.mean([roc_auc_score(y.iloc[v],
        model_cls(**params).fit(X_in[t], y.iloc[t]).predict_proba(X_in[v])[:,1])
        for t, v in inner.split(X_in, y)])

print("\nOptimizing LightGBM GBDT (50 trials)...")
def lgb_obj(trial):
    return inner_auc(lgb.LGBMClassifier, dict(
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
sl.enqueue_trial(PREV_LGB)
sl.optimize(lgb_obj, n_trials=50, show_progress_bar=False)
best_lgb = sl.best_params
print(f"  Best LGB GBDT: {sl.best_value:.4f}")

print("\nOptimizing XGBoost (50 trials)...")
def xgb_obj(trial):
    return inner_auc(xgb.XGBClassifier, dict(
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
sx.enqueue_trial(PREV_XGB)
sx.optimize(xgb_obj, n_trials=50, show_progress_bar=False)
best_xgb = sx.best_params
print(f"  Best XGB: {sx.best_value:.4f}")

print("\nOptimizing CatBoost (50 trials)...")
def cat_obj(trial):
    return inner_auc(cb.CatBoostClassifier, dict(
        iterations=trial.suggest_int('iterations',500,3000),
        learning_rate=trial.suggest_float('learning_rate',0.005,0.1,log=True),
        depth=trial.suggest_int('depth',4,8),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg',0.5,10.0),
        subsample=trial.suggest_float('subsample',0.6,1.0),
        colsample_bylevel=trial.suggest_float('colsample_bylevel',0.5,1.0),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf',5,30),
        random_seed=BASE_SEED, verbose=0))
sc = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
sc.optimize(cat_obj, n_trials=50, show_progress_bar=False)
best_cat = sc.best_params
print(f"  Best CAT: {sc.best_value:.4f}")

print("\nOptimizing MLP (50 trials)...")
def mlp_obj(trial):
    n1 = trial.suggest_int('n1', 64, 512)
    n2 = trial.suggest_int('n2', 32, 256)
    n3 = trial.suggest_int('n3', 16, 128)
    return inner_auc(MLPClassifier, dict(
        hidden_layer_sizes=(n1, n2, n3),
        activation=trial.suggest_categorical('activation', ['relu', 'tanh']),
        alpha=trial.suggest_float('alpha', 1e-4, 1.0, log=True),
        learning_rate_init=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        max_iter=300, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15,
        random_state=BASE_SEED), X_in=X_scaled)
sm = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
sm.optimize(mlp_obj, n_trials=50, show_progress_bar=False)
best_mlp_p = sm.best_params
best_mlp   = dict(
    hidden_layer_sizes=(best_mlp_p['n1'], best_mlp_p['n2'], best_mlp_p['n3']),
    activation=best_mlp_p['activation'],
    alpha=best_mlp_p['alpha'],
    learning_rate_init=best_mlp_p['lr'],
    max_iter=300, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=15,
)
print(f"  Best MLP: {sm.best_value:.4f}  layers={best_mlp['hidden_layer_sizes']}")

# ── Model factory ─────────────────────────────────────────────────────────────
DART_LGB = dict(**best_lgb, boosting_type='dart',
                drop_rate=0.1, skip_drop=0.5,
                verbose=-1, n_jobs=-1)

def make_models(seed):
    return {
        'xgb' : xgb.XGBClassifier(**best_xgb, eval_metric='auc',
                    random_state=seed, tree_method='hist', n_jobs=-1),
        'lgb' : lgb.LGBMClassifier(**best_lgb, random_state=seed,
                    verbose=-1, n_jobs=-1),
        'lgb_dart': lgb.LGBMClassifier(**DART_LGB, random_state=seed),
        'cat' : cb.CatBoostClassifier(**best_cat, random_seed=seed, verbose=0),
        'mlp' : MLPClassifier(**best_mlp, random_state=seed),
    }

MODEL_NAMES = list(make_models(BASE_SEED).keys())
MLP_MODELS  = {'mlp'}

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
            Xtr = X_scaled[tr_idx] if nm in MLP_MODELS else X_arr[tr_idx]
            Xvl = X_scaled[val_idx] if nm in MLP_MODELS else X_arr[val_idx]
            Xte = X_test_scaled     if nm in MLP_MODELS else X_test_arr
            m   = clone(model)
            m.fit(Xtr, yf_tr)
            seed_oof[nm][val_idx]  = m.predict_proba(Xvl)[:, 1]
            seed_tst[nm]          += m.predict_proba(Xte)[:, 1] / N_FOLDS
            print(f"  {nm}={roc_auc_score(yf_val, seed_oof[nm][val_idx]):.4f}", end="")
        print()
    seed_aucs = {nm: roc_auc_score(y, seed_oof[nm]) for nm in MODEL_NAMES}
    print(f"  seed OOF: {' '.join(f'{nm}={v:.4f}' for nm,v in seed_aucs.items())}")
    for nm in MODEL_NAMES:
        oof_acc[nm]  += seed_oof[nm]  / len(SEEDS)
        test_acc[nm] += seed_tst[nm]  / len(SEEDS)

print("\n── Base model OOF AUC (avg across seeds) ───────────────")
oof_aucs = {nm: roc_auc_score(y, oof_acc[nm]) for nm in MODEL_NAMES}
for nm, auc in oof_aucs.items():
    print(f"  {nm:10s}: {auc:.4f}")

# ── LGB stacking meta-learner ─────────────────────────────────────────────────
# Input: OOF base-model predictions + 8 key raw features
KEY_FEATS = ['Age_qrank', 'Balance_qrank', 'GeoTargetEnc', 'SurnameTargetEnc',
             'Prod3or4', 'Prod1', 'IsActiveMember', 'SeniorCustomer']
key_idx   = [list(X.columns).index(f) for f in KEY_FEATS]

oof_meta_feats  = np.hstack([
    np.column_stack([oof_acc[nm] for nm in MODEL_NAMES]),
    X_arr[:, key_idx]
])
test_meta_feats = np.hstack([
    np.column_stack([test_acc[nm] for nm in MODEL_NAMES]),
    X_test_arr[:, key_idx]
])

print("\nTraining LGB meta-learner (5-fold nested CV)...")
meta_lgb_params = dict(
    n_estimators=300, learning_rate=0.05, max_depth=3,
    num_leaves=15, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    random_state=BASE_SEED, verbose=-1, n_jobs=-1,
)
meta_cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
meta_oof    = np.zeros(n_train)
meta_test   = np.zeros(len(test))
for tr, val in meta_cv.split(oof_meta_feats, y):
    m = lgb.LGBMClassifier(**meta_lgb_params)
    m.fit(oof_meta_feats[tr], y.iloc[tr])
    meta_oof[val]  = m.predict_proba(oof_meta_feats[val])[:, 1]
    meta_test     += m.predict_proba(test_meta_feats)[:, 1] / N_FOLDS
meta_auc = roc_auc_score(y, meta_oof)
print(f"  Meta-learner OOF AUC: {meta_auc:.4f}")

# ── Final blend: base weighted avg + meta-learner ─────────────────────────────
w         = np.array([oof_aucs[nm] for nm in MODEL_NAMES]); w /= w.sum()
base_oof  = np.column_stack([oof_acc[nm]  for nm in MODEL_NAMES]) @ w
base_test = np.column_stack([test_acc[nm] for nm in MODEL_NAMES]) @ w
base_auc  = roc_auc_score(y, base_oof)
print(f"  Base weighted avg OOF AUC: {base_auc:.4f}")

# Blend base + meta (weight toward meta if it's better)
for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blend_auc = roc_auc_score(y, alpha * meta_oof + (1-alpha) * base_oof)
    print(f"  blend alpha={alpha}: {blend_auc:.4f}")

best_alpha = max([0.3, 0.4, 0.5, 0.6, 0.7],
                 key=lambda a: roc_auc_score(y, a*meta_oof + (1-a)*base_oof))
final_oof  = best_alpha * meta_oof  + (1 - best_alpha) * base_oof
final_test = best_alpha * meta_test + (1 - best_alpha) * base_test
print(f"\nFinal blend (alpha={best_alpha}) OOF AUC: {roc_auc_score(y, final_oof):.4f}")

# ── Save submission ────────────────────────────────────────────────────────────
sub = pd.read_csv('sample_submission.csv')
sub['Exited'] = final_test
sub.to_csv('submission.csv', index=False)
print(f"\nDone! submission.csv saved.")
print(sub.head())
