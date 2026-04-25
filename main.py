import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

SEED    = 42
N_FOLDS = 10   # more folds → lower variance OOF

# ── Load data ─────────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
y     = train['Exited'].astype(int)
n_train     = len(train)
global_mean = y.mean()

# ── Smoothed (Bayesian) OOF target encoding ───────────────────────────────────
def smooth_te_oof(s_train, y_train, s_test, alpha=30):
    """Bayesian-smoothed OOF target encoding. alpha controls shrinkage to global mean."""
    gm = y_train.mean()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(s_train), gm)
    for tr_idx, val_idx in cv.split(s_train, y_train):
        stats = (pd.DataFrame({'v': s_train.iloc[tr_idx].values,
                               't': y_train.iloc[tr_idx].values})
                   .groupby('v')['t'].agg(['count', 'mean']))
        smap = ((stats['count'] * stats['mean'] + alpha * gm) /
                (stats['count'] + alpha)).to_dict()
        oof[val_idx] = s_train.iloc[val_idx].map(smap).fillna(gm).values
    # full-train map for test
    stats = (pd.DataFrame({'v': s_train.values, 't': y_train.values})
               .groupby('v')['t'].agg(['count', 'mean']))
    fmap = ((stats['count'] * stats['mean'] + alpha * gm) /
            (stats['count'] + alpha)).to_dict()
    test_enc = s_test.map(fmap).fillna(gm).values
    return oof, test_enc, fmap

surname_oof,  surname_test,  surname_fmap  = smooth_te_oof(train['Surname'],   y, test['Surname'],   alpha=30)
geo_oof,      geo_test,      geo_fmap      = smooth_te_oof(train['Geography'], y, test['Geography'], alpha=5)
gender_oof,   gender_test,   gender_fmap   = smooth_te_oof(train['Gender'],    y, test['Gender'],    alpha=5)

# ── Feature engineering ───────────────────────────────────────────────────────
all_surnames     = pd.concat([train['Surname'], test['Surname']])
surname_freq_map = all_surnames.value_counts().to_dict()

def engineer(df, sur_enc, geo_enc, gen_enc):
    df = df.copy()
    df['Gender_bin']  = (df['Gender'] == 'Male').astype(int)
    df['Geo_France']  = (df['Geography'] == 'France').astype(int)
    df['Geo_Germany'] = (df['Geography'] == 'Germany').astype(int)
    df['Geo_Spain']   = (df['Geography'] == 'Spain').astype(int)

    df['SurnameTargetEnc'] = sur_enc
    df['GeoTargetEnc']     = geo_enc
    df['GenderTargetEnc']  = gen_enc
    df['SurnameFreq']      = df['Surname'].map(surname_freq_map).fillna(1)

    # Product flags (huge signal: 3/4 → ~96% churn)
    df['Prod1']        = (df['NumOfProducts'] == 1).astype(int)
    df['Prod2']        = (df['NumOfProducts'] == 2).astype(int)
    df['Prod3or4']     = (df['NumOfProducts'] >= 3).astype(int)
    df['ProdHighRisk'] = ((df['NumOfProducts'] == 1) | (df['NumOfProducts'] >= 3)).astype(int)

    # Balance
    df['HasBalance']      = (df['Balance'] > 0).astype(int)
    df['HighBalance']     = (df['Balance'] > 100_000).astype(int)
    df['VeryHighBalance'] = (df['Balance'] > 150_000).astype(int)
    df['LogBalance']      = np.log1p(df['Balance'])

    # Age
    df['SeniorCustomer'] = (df['Age'] > 50).astype(int)
    df['MidAge']         = ((df['Age'] >= 35) & (df['Age'] <= 50)).astype(int)
    df['Age2']           = df['Age'] ** 2
    df['AgeGroup']       = pd.cut(
        df['Age'], bins=[0,25,30,35,40,45,50,55,60,100], labels=False).fillna(0)

    # Ratios
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['CreditScorePerAge']  = df['CreditScore'] / (df['Age'] + 1)
    df['ProductsPerTenure']  = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['TenureAgeRatio']     = df['Tenure'] / (df['Age'] + 1)
    df['BalancePerProduct']  = df['Balance'] / (df['NumOfProducts'] + 1)
    df['SalaryPerAge']       = df['EstimatedSalary'] / (df['Age'] + 1)
    df['LogCreditScore']     = np.log1p(df['CreditScore'])
    df['LogSalary']          = np.log1p(df['EstimatedSalary'])

    # Interactions
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
    df['GeoEnc_x_Age']      = geo_enc * df['Age']
    df['GeoEnc_x_Balance']  = geo_enc * df['Balance']
    df['SurnameEnc_x_Prod'] = sur_enc * df['NumOfProducts']
    df['Tenure_is_0']       = (df['Tenure'] == 0).astype(int)
    df['AgeProd1']          = df['Age'] * df['Prod1']
    df['BalanceProd']       = df['Balance'] * df['NumOfProducts']

    drop_cols = ['id', 'CustomerId', 'Surname', 'Geography', 'Gender']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

X      = engineer(train.drop(columns=['Exited']), surname_oof,  geo_oof,  gender_oof)
X_test = engineer(test,                            surname_test, geo_test, gender_test)
X      = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
print(f"Train shape: {X.shape} | Churn rate: {y.mean():.1%}")

X_arr      = X.values.astype(float)
X_test_arr = X_test.values.astype(float)

# ── Optuna HPO for LightGBM (warm-start from previous best) ──────────────────
print("\nOptimizing LightGBM (75 trials)...")
PREV_LGB = dict(n_estimators=1171, learning_rate=0.005697794581250653,
                max_depth=4, num_leaves=53, subsample=0.6707262232640725,
                colsample_bytree=0.5780769964624514, min_child_samples=30,
                reg_alpha=0.14041249938727385, reg_lambda=0.008793362023805958)

def lgb_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 500, 4000),
        learning_rate     = trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
        max_depth         = trial.suggest_int('max_depth', 3, 8),
        num_leaves        = trial.suggest_int('num_leaves', 20, 120),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        min_child_samples = trial.suggest_int('min_child_samples', 10, 60),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        random_state=SEED, verbose=-1, n_jobs=-1,
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = [roc_auc_score(y.iloc[v],
              lgb.LGBMClassifier(**p).fit(X_arr[t], y.iloc[t]).predict_proba(X_arr[v])[:,1])
              for t, v in inner_cv.split(X_arr, y)]
    return np.mean(scores)

sl = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
sl.enqueue_trial(PREV_LGB)
sl.optimize(lgb_obj, n_trials=75, show_progress_bar=False)
best_lgb = sl.best_params
print(f"  Best LGB: {sl.best_value:.4f}")

# ── Optuna HPO for XGBoost ────────────────────────────────────────────────────
print("\nOptimizing XGBoost (75 trials)...")
PREV_XGB = dict(n_estimators=887, learning_rate=0.01164177793000955,
                max_depth=3, subsample=0.6562323205810695,
                colsample_bytree=0.9251313643434012, min_child_weight=5,
                gamma=0.2087124905867131, reg_alpha=0.002007689708810798,
                reg_lambda=0.31870743101267013)

def xgb_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 500, 4000),
        learning_rate     = trial.suggest_float('learning_rate', 0.003, 0.05, log=True),
        max_depth         = trial.suggest_int('max_depth', 3, 7),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 1.0),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 15),
        gamma             = trial.suggest_float('gamma', 0.0, 0.5),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        eval_metric='auc', random_state=SEED, tree_method='hist', n_jobs=-1,
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = [roc_auc_score(y.iloc[v],
              xgb.XGBClassifier(**p).fit(X_arr[t], y.iloc[t]).predict_proba(X_arr[v])[:,1])
              for t, v in inner_cv.split(X_arr, y)]
    return np.mean(scores)

sx = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
sx.enqueue_trial(PREV_XGB)
sx.optimize(xgb_obj, n_trials=75, show_progress_bar=False)
best_xgb = sx.best_params
print(f"  Best XGB: {sx.best_value:.4f}")

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    'xgb': xgb.XGBClassifier(
        **best_xgb, eval_metric='auc', random_state=SEED, tree_method='hist', n_jobs=-1),
    'lgb': lgb.LGBMClassifier(
        **best_lgb, random_state=SEED, verbose=-1, n_jobs=-1),
    'cat': cb.CatBoostClassifier(
        iterations=2000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
        subsample=0.8, colsample_bylevel=0.7, min_data_in_leaf=15,
        eval_metric='AUC', random_seed=SEED, verbose=0),
}

# ── OOF function ──────────────────────────────────────────────────────────────
def run_oof(X_tr_arr, y_tr, X_te_arr, models_dict):
    cv    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n     = len(y_tr)
    oofs  = {nm: np.zeros(n)               for nm in models_dict}
    tests = {nm: np.zeros(len(X_te_arr))   for nm in models_dict}

    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(X_tr_arr, y_tr), 1):
        Xf_tr, Xf_val = X_tr_arr[tr_idx], X_tr_arr[val_idx]
        yf_tr, yf_val = y_tr.iloc[tr_idx].values, y_tr.iloc[val_idx].values
        print(f"  fold {fold_i:02d}", end="")
        for nm, model in models_dict.items():
            m = clone(model)
            m.fit(Xf_tr, yf_tr)
            oofs[nm][val_idx] = m.predict_proba(Xf_val)[:, 1]
            tests[nm] += m.predict_proba(X_te_arr)[:, 1] / N_FOLDS
            print(f"  {nm}={roc_auc_score(yf_val, oofs[nm][val_idx]):.4f}", end="")
        print()
    return oofs, tests

# ── Round 0: train on original data ──────────────────────────────────────────
print("\n── Round 0: original training data ─────────────────────")
oof_preds, test_preds = run_oof(X_arr, y, X_test_arr, models)

print("\nOOF AUC (round 0):")
oof_aucs = {nm: roc_auc_score(y, oof_preds[nm]) for nm in models}
for nm, auc in oof_aucs.items():
    print(f"  {nm}: {auc:.4f}")

# Rank-averaged ensemble test prediction
def rank_ensemble(preds_dict, n_test):
    return np.column_stack([
        rankdata(preds_dict[nm]) / n_test for nm in preds_dict
    ]).mean(axis=1)

test_ensemble_r0 = rank_ensemble(test_preds, len(test))

# ── Pseudo-labeling ───────────────────────────────────────────────────────────
def pseudo_label_round(X_orig, y_orig, X_te_arr, test_pred_probs,
                       hi=0.85, lo=0.15, rnd=1):
    """Add high-confidence test predictions as extra training samples."""
    mask_pos = test_pred_probs > hi
    mask_neg = test_pred_probs < lo
    n_pos = mask_pos.sum(); n_neg = mask_neg.sum()
    print(f"\n── Pseudo-label round {rnd}: +{n_pos} churn / +{n_neg} non-churn "
          f"(thresholds {hi}/{lo}) ─────")
    if n_pos + n_neg == 0:
        return X_orig, y_orig

    X_pseudo = np.concatenate([X_orig, X_te_arr[mask_pos | mask_neg]])
    y_pseudo  = pd.Series(np.concatenate([
        y_orig.values,
        np.ones(n_pos, dtype=int),
        np.zeros(n_neg, dtype=int),
    ]))
    # interleave pos and neg indices correctly
    y_pseudo = pd.Series(np.concatenate([
        y_orig.values,
        np.ones(n_pos,  dtype=int),
        np.zeros(n_neg, dtype=int),
    ]))
    # need to rebuild: pos first then neg may not match mask order → reindex
    pseudo_labels = np.where(test_pred_probs[mask_pos | mask_neg] > hi, 1, 0)
    y_pseudo = pd.Series(np.concatenate([y_orig.values, pseudo_labels]))
    return X_pseudo, y_pseudo

# Round 1
X_pl1, y_pl1 = pseudo_label_round(
    X_arr, y, X_test_arr, test_ensemble_r0, hi=0.85, lo=0.15, rnd=1)
oof_pl1, test_pl1 = run_oof(X_pl1, y_pl1, X_test_arr, models)
# evaluate only on original training rows
oof_aucs_pl1 = {nm: roc_auc_score(y, oof_pl1[nm][:n_train]) for nm in models}
print("OOF AUC (pseudo round 1, original rows):")
for nm, auc in oof_aucs_pl1.items():
    print(f"  {nm}: {auc:.4f}")

test_ensemble_r1 = rank_ensemble(test_pl1, len(test))

# Round 2
X_pl2, y_pl2 = pseudo_label_round(
    X_arr, y, X_test_arr, test_ensemble_r1, hi=0.80, lo=0.20, rnd=2)
oof_pl2, test_pl2 = run_oof(X_pl2, y_pl2, X_test_arr, models)
oof_aucs_pl2 = {nm: roc_auc_score(y, oof_pl2[nm][:n_train]) for nm in models}
print("OOF AUC (pseudo round 2, original rows):")
for nm, auc in oof_aucs_pl2.items():
    print(f"  {nm}: {auc:.4f}")

test_ensemble_r2 = rank_ensemble(test_pl2, len(test))

# ── Pick best round ───────────────────────────────────────────────────────────
best_r0  = np.mean(list(oof_aucs.values()))
best_r1  = np.mean(list(oof_aucs_pl1.values()))
best_r2  = np.mean(list(oof_aucs_pl2.values()))
print(f"\nMean OOF: r0={best_r0:.4f}  r1={best_r1:.4f}  r2={best_r2:.4f}")

if best_r2 >= best_r1 and best_r2 >= best_r0:
    final_test_preds = test_pl2
    best_oof_aucs    = oof_aucs_pl2
    print("→ Using pseudo-label round 2")
elif best_r1 >= best_r0:
    final_test_preds = test_pl1
    best_oof_aucs    = oof_aucs_pl1
    print("→ Using pseudo-label round 1")
else:
    final_test_preds = test_preds
    best_oof_aucs    = oof_aucs
    print("→ Using original (pseudo-labeling did not help)")

# ── Final ensemble: rank average + meta-learner (pick best) ──────────────────
# Re-compute OOF stack on original data only for meta-learner
oof_stack  = np.column_stack([oof_preds[nm] for nm in models])   # round 0 oofs
test_stack = np.column_stack([final_test_preds[nm] for nm in models])

meta = LogisticRegression(C=1.0, max_iter=1000)
meta.fit(oof_stack, y)
meta_oof  = meta.predict_proba(oof_stack)[:, 1]
meta_test = meta.predict_proba(test_stack)[:, 1]
meta_auc  = roc_auc_score(y, meta_oof)

rank_test = rank_ensemble(final_test_preds, len(test))
w         = np.array([best_oof_aucs[nm] for nm in models])
w         = w / w.sum()
w_test    = test_stack @ w
w_auc     = roc_auc_score(y, oof_stack @ np.array([oof_aucs[nm] for nm in models]) /
                              np.array([oof_aucs[nm] for nm in models]).sum())

print(f"\nMeta-learner OOF AUC : {meta_auc:.4f}")
print(f"Weighted avg OOF AUC : {w_auc:.4f}")

# Blend meta + weighted
final = 0.5 * meta_test + 0.5 * w_test
blend_oof = 0.5 * meta_oof + 0.5 * (oof_stack @ np.array([oof_aucs[nm] for nm in models]) /
                                      np.array([oof_aucs[nm] for nm in models]).sum())
print(f"Blend OOF AUC        : {roc_auc_score(y, blend_oof):.4f}")

# ── Save submission ────────────────────────────────────────────────────────────
sub = pd.read_csv('sample_submission.csv')
sub['Exited'] = final
sub.to_csv('submission.csv', index=False)
print(f"\nDone! submission.csv saved.")
print(sub.head())
