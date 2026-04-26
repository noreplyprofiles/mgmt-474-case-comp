"""
v5 — multi-seed ensemble + MLP diversity
  • 4 random seeds × (XGB + LGB + CAT + MLP) = 16 models averaged
  • Optuna warm-started for XGB / LGB / CAT (50 trials each)
  • MLP with scaled features adds genuine diversity
  • Raw probability averaging (consistent OOF & test)
  • Raw OOF target encoding, 5-fold CV
  • No pseudo-labeling, no smoothing
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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

# ── Raw OOF target encoding ───────────────────────────────────────────────────
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
from scipy.stats import rankdata
rank_all = {c: rankdata(all_data[c].values) / len(all_data) for c in RANK_COLS}

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

# Scaled versions for MLP
scaler       = StandardScaler()
X_scaled     = scaler.fit_transform(X_arr)
X_test_scaled= scaler.transform(X_test_arr)

print(f"Train shape: {X.shape} | Churn rate: {y.mean():.1%}")

# ── Optuna HPO (warm-started) ─────────────────────────────────────────────────
PREV_LGB = dict(n_estimators=1171, learning_rate=0.005697794581250653,
                max_depth=4, num_leaves=53, subsample=0.6707262232640725,
                colsample_bytree=0.5780769964624514, min_child_samples=30,
                reg_alpha=0.14041249938727385, reg_lambda=0.008793362023805958)
PREV_XGB = dict(n_estimators=887, learning_rate=0.01164177793000955,
                max_depth=3, subsample=0.6562323205810695,
                colsample_bytree=0.9251313643434012, min_child_weight=5,
                gamma=0.2087124905867131, reg_alpha=0.002007689708810798,
                reg_lambda=0.31870743101267013)

def inner_cv_auc(model_cls, params, X_in=None, seed_offset=0):
    if X_in is None: X_in = X_arr
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=BASE_SEED + seed_offset)
    return np.mean([
        roc_auc_score(y.iloc[v],
            model_cls(**params).fit(X_in[t], y.iloc[t]).predict_proba(X_in[v])[:,1])
        for t, v in inner.split(X_in, y)
    ])

print("\nOptimizing LightGBM (50 trials)...")
def lgb_obj(trial):
    return inner_cv_auc(lgb.LGBMClassifier, dict(
        n_estimators      = trial.suggest_int('n_estimators', 500, 4000),
        learning_rate     = trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
        max_depth         = trial.suggest_int('max_depth', 3, 8),
        num_leaves        = trial.suggest_int('num_leaves', 20, 120),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        min_child_samples = trial.suggest_int('min_child_samples', 5, 60),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        random_state=BASE_SEED, verbose=-1, n_jobs=-1,
    ))
sl = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
sl.enqueue_trial(PREV_LGB)
sl.optimize(lgb_obj, n_trials=50, show_progress_bar=False)
best_lgb = sl.best_params
print(f"  Best LGB: {sl.best_value:.4f}  {best_lgb}")

print("\nOptimizing XGBoost (50 trials)...")
def xgb_obj(trial):
    return inner_cv_auc(xgb.XGBClassifier, dict(
        n_estimators      = trial.suggest_int('n_estimators', 500, 4000),
        learning_rate     = trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
        max_depth         = trial.suggest_int('max_depth', 3, 7),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 15),
        gamma             = trial.suggest_float('gamma', 0.0, 0.5),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        eval_metric='auc', random_state=BASE_SEED, tree_method='hist', n_jobs=-1,
    ))
sx = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
sx.enqueue_trial(PREV_XGB)
sx.optimize(xgb_obj, n_trials=50, show_progress_bar=False)
best_xgb = sx.best_params
print(f"  Best XGB: {sx.best_value:.4f}  {best_xgb}")

print("\nOptimizing CatBoost (50 trials)...")
def cat_obj(trial):
    return inner_cv_auc(cb.CatBoostClassifier, dict(
        iterations        = trial.suggest_int('iterations', 500, 3000),
        learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        depth             = trial.suggest_int('depth', 4, 8),
        l2_leaf_reg       = trial.suggest_float('l2_leaf_reg', 0.5, 10.0),
        subsample         = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        min_data_in_leaf  = trial.suggest_int('min_data_in_leaf', 5, 30),
        random_seed=BASE_SEED, verbose=0,
    ))
sc = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
sc.optimize(cat_obj, n_trials=50, show_progress_bar=False)
best_cat = sc.best_params
print(f"  Best CAT: {sc.best_value:.4f}  {best_cat}")

# ── Model factory (seed-parameterized) ───────────────────────────────────────
def make_models(seed):
    return {
        'xgb': xgb.XGBClassifier(
            **best_xgb, eval_metric='auc',
            random_state=seed, tree_method='hist', n_jobs=-1),
        'lgb': lgb.LGBMClassifier(
            **best_lgb, random_state=seed, verbose=-1, n_jobs=-1),
        'cat': cb.CatBoostClassifier(
            **best_cat, random_seed=seed, verbose=0),
        'mlp': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu', solver='adam',
            alpha=0.01, learning_rate_init=0.001,
            max_iter=300, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20,
            random_state=seed),
    }

MODEL_NAMES = list(make_models(BASE_SEED).keys())

# ── Multi-seed OOF training ───────────────────────────────────────────────────
# Accumulate OOF and test predictions across all seeds
oof_acc  = {nm: np.zeros(n_train)   for nm in MODEL_NAMES}
test_acc = {nm: np.zeros(len(test)) for nm in MODEL_NAMES}

print(f"\n── Multi-seed OOF ({len(SEEDS)} seeds × {N_FOLDS} folds × {len(MODEL_NAMES)} models) ──────────")
for seed in SEEDS:
    print(f"\n  seed={seed}")
    models    = make_models(seed)
    cv        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_oof  = {nm: np.zeros(n_train)   for nm in MODEL_NAMES}
    seed_test = {nm: np.zeros(len(test)) for nm in MODEL_NAMES}

    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_arr, y), 1):
        Xf_tr  = X_arr[tr_idx];    Xf_val  = X_arr[val_idx]
        Xs_tr  = X_scaled[tr_idx]; Xs_val  = X_scaled[val_idx]
        yf_tr  = y.iloc[tr_idx].values
        yf_val = y.iloc[val_idx].values
        print(f"    fold {fold_i}", end="")
        for nm, model in models.items():
            Xtr = Xs_tr if nm == 'mlp' else Xf_tr
            Xvl = Xs_val if nm == 'mlp' else Xf_val
            Xte = X_test_scaled if nm == 'mlp' else X_test_arr
            m   = clone(model)
            m.fit(Xtr, yf_tr)
            seed_oof[nm][val_idx]  = m.predict_proba(Xvl)[:, 1]
            seed_test[nm]         += m.predict_proba(Xte)[:, 1] / N_FOLDS
            print(f"  {nm}={roc_auc_score(yf_val, seed_oof[nm][val_idx]):.4f}", end="")
        print()

    seed_aucs = {nm: roc_auc_score(y, seed_oof[nm]) for nm in MODEL_NAMES}
    print(f"  OOF: {' | '.join(f'{nm}={seed_aucs[nm]:.4f}' for nm in MODEL_NAMES)}")
    for nm in MODEL_NAMES:
        oof_acc[nm]  += seed_oof[nm]  / len(SEEDS)
        test_acc[nm] += seed_test[nm] / len(SEEDS)

# ── Final ensemble ────────────────────────────────────────────────────────────
print("\n── Final OOF AUC (averaged across seeds) ────────────────")
oof_aucs = {}
for nm in MODEL_NAMES:
    oof_aucs[nm] = roc_auc_score(y, oof_acc[nm])
    print(f"  {nm:5s}: {oof_aucs[nm]:.4f}")

# Weighted average by OOF AUC
w          = np.array([oof_aucs[nm] for nm in MODEL_NAMES])
w          = w / w.sum()
test_stack = np.column_stack([test_acc[nm] for nm in MODEL_NAMES])
oof_stack  = np.column_stack([oof_acc[nm]  for nm in MODEL_NAMES])
final_test = test_stack @ w
final_oof  = oof_stack  @ w
print(f"\nWeighted ensemble OOF AUC: {roc_auc_score(y, final_oof):.4f}")
print(f"Weights: {dict(zip(MODEL_NAMES, w.round(3)))}")

# ── Save submission ────────────────────────────────────────────────────────────
sub = pd.read_csv('sample_submission.csv')
sub['Exited'] = final_test
sub.to_csv('submission.csv', index=False)
print(f"\nDone! submission.csv saved.")
print(sub.head())
