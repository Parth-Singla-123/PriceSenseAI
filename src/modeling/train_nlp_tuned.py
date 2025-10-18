import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import optuna
import os
import joblib
from functools import partial

# --- Configuration ---
CONFIG = {
    'PROCESSED_DATA_DIR': 'processed_data/',
    'MODEL_OUTPUT_DIR': 'models/',
    'SUBMISSION_OUTPUT_DIR': 'submissions/',
    'EMBEDDING_FILE': 'text_embeddings.npz', # <-- INPUT FILE
    'N_SPLITS': 5,
    'RANDOM_STATE': 42,
    'OPTUNA_N_TRIALS': 15,
    'BEST_LGBM_PARAMS': None
}

# ... (The rest of the script is exactly the same as the final version from the previous step) ...
# It starts by loading the data and then IMMEDIATELY loads the embeddings file.
# It does NOT contain any SentenceTransformer code.

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

def objective(trial, train_embeddings, y_train):
    params = {
        'objective': 'huber', 'metric': 'mae', 'random_state': CONFIG['RANDOM_STATE'],
        'n_estimators': 2000, 'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 5.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 5.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
    }

    kf = KFold(n_splits=CONFIG['N_SPLITS'], shuffle=True, random_state=CONFIG['RANDOM_STATE'])
    oof_maes = []
    for _, (train_idx, val_idx) in enumerate(kf.split(train_embeddings)):
        X_train_fold, X_val_fold = train_embeddings[train_idx], train_embeddings[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = lgb.LGBMRegressor(**params); model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(100, verbose=False)])
        val_preds = model.predict(X_val_fold); oof_maes.append(np.mean(np.abs(y_val_fold - val_preds)))
    return np.mean(oof_maes)

if __name__ == '__main__':
    # --- 1. Load Data and PRE-COMPUTED Embeddings ---
    print("Loading data and pre-computed embeddings...")
    df_train = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features.csv'))
    df_test = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features_test.csv'))
    
    try:
        embeddings = np.load(os.path.join(CONFIG['PROCESSED_DATA_DIR'], CONFIG['EMBEDDING_FILE']))
    except FileNotFoundError:
        print(f"Error: Embedding file not found at {os.path.join(CONFIG['PROCESSED_DATA_DIR'], CONFIG['EMBEDDING_FILE'])}")
        print("Please run the 'generate_embeddings.py' script first.")
        exit()

    train_embeddings = embeddings['train']
    test_embeddings = embeddings['test']
    target = 'price_log1p'
    y_train = df_train[target]

    # --- The rest of the script (Tuning, Training, Saving) is exactly the same ---
    print(f"Starting hyperparameter tuning for NLP model with {CONFIG['OPTUNA_N_TRIALS']} trials..."); 
    objective_with_args = partial(objective, train_embeddings=train_embeddings, y_train=y_train)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_with_args, n_trials=CONFIG['OPTUNA_N_TRIALS'])
    
    CONFIG['BEST_LGBM_PARAMS'] = {**study.best_params, 'objective': 'huber', 'metric': 'mae', 'random_state': CONFIG['RANDOM_STATE'], 'n_estimators': 2000, 'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': -1}
    print("Best MAE from study:", study.best_value)
    print("Best parameters found:", study.best_params)

    print("\nTraining final NLP model with best parameters..."); 
    kf = KFold(n_splits=CONFIG['N_SPLITS'], shuffle=True, random_state=CONFIG['RANDOM_STATE'])
    oof_preds = np.zeros(df_train.shape[0]); test_preds = np.zeros(df_test.shape[0])

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_embeddings)):
        print(f"--- Fold {fold+1}/{CONFIG['N_SPLITS']} ---"); X_train_fold, X_val_fold = train_embeddings[train_idx], train_embeddings[val_idx]; y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = lgb.LGBMRegressor(**CONFIG['BEST_LGBM_PARAMS']); model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(100, verbose=False)])
        val_preds = model.predict(X_val_fold); oof_preds[val_idx] = val_preds; test_preds += model.predict(test_embeddings) / CONFIG['N_SPLITS']
        joblib.dump(model, os.path.join(CONFIG['MODEL_OUTPUT_DIR'], f'nlp_lgbm_model_fold_{fold}.pkl'))
        
    overall_smape = smape(np.expm1(y_train), np.expm1(oof_preds))
    print(f"\nOverall OOF SMAPE for TUNED NLP LGBM Model: {overall_smape:.4f}")
    
    oof_df = pd.DataFrame({'sample_id': df_train['sample_id'], 'oof_prediction_nlp_lgbm': np.expm1(oof_preds)})
    oof_df.to_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'nlp_lgbm_model_oof.csv'), index=False)
    print("Tuned NLP LGBM OOF predictions saved.")