import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
import optuna
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial # Cleaner way to pass arguments to Optuna

# --- Configuration ---
CONFIG = {
    'PROCESSED_DATA_DIR': 'processed_data/',
    'MODEL_OUTPUT_DIR': 'models/',
    'SUBMISSION_OUTPUT_DIR': 'submissions/',
    'N_SPLITS': 5,
    'RANDOM_STATE': 42,
    'OPTUNA_N_TRIALS': 15, # Using your requested 15 trials
    'BEST_LGBM_PARAMS': None
}

# Ensure directories exist
os.makedirs(CONFIG['MODEL_OUTPUT_DIR'], exist_ok=True)
os.makedirs(CONFIG['SUBMISSION_OUTPUT_DIR'], exist_ok=True)

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

def create_advanced_brand_features(X_train, X_val, X_test, target):
    # This function is now called by both the objective and the final training loop
    # It requires the 'brand' column to be present in the input dataframes
    
    brand_stats_map = X_train.groupby('brand')[target].agg(['mean', 'std', 'max'])
    brand_stats_map.columns = ['brand_mean_price', 'brand_std_price', 'brand_max_price']
    
    unit_cols = [col for col in X_train.columns if 'unit_' in col]
    X_train['unit_consolidated'] = X_train[unit_cols].idxmax(axis=1)
    brand_unit_map = X_train.groupby(['brand', 'unit_consolidated'])[target].mean().rename('brand_unit_mean_price')

    brand_price_density_map = X_train.groupby('brand').apply(
        lambda g: (np.expm1(g[target]) / g['normalized_quantity'].replace(0, 1)).mean()
    ).rename('brand_price_density')

    global_mean_stats = brand_stats_map.mean()
    global_mean_brand_unit = brand_unit_map.mean()
    global_mean_density = brand_price_density_map.mean()

    all_dfs = {'train': X_train, 'val': X_val, 'test': X_test}
    processed_dfs = {}
    for name, df in all_dfs.items():
        if df.empty: # Guard against empty test dataframe during objective eval
            processed_dfs[name] = df
            continue

        df = df.join(brand_stats_map, on='brand')
        df['unit_consolidated'] = df[unit_cols].idxmax(axis=1)
        df = df.join(brand_unit_map, on=['brand', 'unit_consolidated'])
        df = df.join(brand_price_density_map, on='brand')
        df.fillna({
            'brand_mean_price': global_mean_stats['brand_mean_price'], 'brand_std_price': global_mean_stats['brand_std_price'],
            'brand_max_price': global_mean_stats['brand_max_price'], 'brand_unit_mean_price': global_mean_brand_unit,
            'brand_price_density': global_mean_density
        }, inplace=True)
        df.drop(columns=['unit_consolidated'], inplace=True)
        processed_dfs[name] = df
    return processed_dfs['train'], processed_dfs['val'], processed_dfs['test']

def objective(trial, df_train, base_features, target):
    """Optuna objective function, now robust and error-free."""
    params = {
        'objective': 'huber', 'metric': 'mae', 'random_state': CONFIG['RANDOM_STATE'],
        'n_estimators': 4000, 'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 5e-2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 30, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
    }

    kf = KFold(n_splits=CONFIG['N_SPLITS'], shuffle=True, random_state=CONFIG['RANDOM_STATE'])
    oof_maes = []
    for _, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        # Explicitly copy to prevent any warnings or errors
        X_train, X_val = df_train.iloc[train_idx].copy(), df_train.iloc[val_idx].copy()
        y_train, y_val = X_train[target], X_val[target]
        
        # Create features safely within the trial
        X_train, X_val, _ = create_advanced_brand_features(X_train, X_val, pd.DataFrame(), target)
        
        current_features = base_features + ['brand_mean_price', 'brand_std_price', 'brand_max_price', 'brand_unit_mean_price', 'brand_price_density']
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train[current_features], y_train,
                  eval_set=[(X_val[current_features], y_val)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        val_preds = model.predict(X_val[current_features])
        oof_maes.append(np.mean(np.abs(y_val - val_preds)))
    return np.mean(oof_maes)

def plot_feature_importance(feature_importances, output_path):
    df_imp = pd.DataFrame({'feature': feature_importances.index, 'importance': feature_importances.values})
    df_imp = df_imp.sort_values('importance', ascending=False).head(40)
    plt.figure(figsize=(10, 12)); sns.barplot(x='importance', y='feature', data=df_imp); plt.title('Top 40 Feature Importances')
    plt.tight_layout(); plt.savefig(output_path); plt.close(); print(f"Feature importance plot saved to {output_path}")

if __name__ == '__main__':
    print("Loading data..."); df_train = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features.csv')); df_test = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features_test.csv'))
    print("Scaling numerical features..."); numerical_cols = ['value', 'pack_multiplier', 'normalized_quantity']; scaler = RobustScaler()
    df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols]); df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])
    joblib.dump(scaler, os.path.join(CONFIG['MODEL_OUTPUT_DIR'], 'robust_scaler.pkl'))
    
    # Base features are defined here, WITHOUT the brand-related ones we create in the loop
    base_features = [col for col in df_train.columns if col not in ['sample_id', 'image_link', 'text_for_embedding', 'price', 'price_log1p', 'brand']]
    target = 'price_log1p'
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

    print(f"Starting hyperparameter tuning with {CONFIG['OPTUNA_N_TRIALS']} trials..."); study = optuna.create_study(direction='minimize')
    # Use functools.partial for a cleaner way to pass arguments
    objective_with_args = partial(objective, df_train=df_train, base_features=base_features, target=target)
    study.optimize(objective_with_args, n_trials=CONFIG['OPTUNA_N_TRIALS'])
    
    CONFIG['BEST_LGBM_PARAMS'] = {**study.best_params, 'objective': 'huber', 'metric': 'mae', 'random_state': CONFIG['RANDOM_STATE'], 'n_estimators': 5000, 'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': -1}
    joblib.dump(study, os.path.join(CONFIG['MODEL_OUTPUT_DIR'], 'optuna_study.pkl')); print("Best MAE from study:", study.best_value); print("Best parameters found:", study.best_params)

    print("\nTraining final model with best parameters..."); kf = KFold(n_splits=CONFIG['N_SPLITS'], shuffle=True, random_state=CONFIG['RANDOM_STATE'])
    oof_preds = np.zeros(df_train.shape[0]); test_preds = np.zeros(df_test.shape[0])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        print(f"--- Fold {fold+1}/{CONFIG['N_SPLITS']} ---"); X_train, X_val = df_train.iloc[train_idx].copy(), df_train.iloc[val_idx].copy(); X_test_fold = df_test.copy()
        y_train, y_val = X_train[target], X_val[target]
        print("Creating advanced features for this fold..."); X_train, X_val, X_test_fold = create_advanced_brand_features(X_train, X_val, X_test_fold, target)
        current_features = [f for f in X_train.columns if f not in ['sample_id', 'image_link', 'text_for_embedding', 'price', 'price_log1p', 'brand']]
        if fold == 0: feature_importances = pd.DataFrame(index=current_features)
        
        model = lgb.LGBMRegressor(**CONFIG['BEST_LGBM_PARAMS']); model.fit(X_train[current_features], y_train, eval_set=[(X_val[current_features], y_val)], callbacks=[lgb.early_stopping(150, verbose=False)])
        val_preds = model.predict(X_val[current_features]); oof_preds[val_idx] = val_preds; test_preds += model.predict(X_test_fold[current_features]) / CONFIG['N_SPLITS']
        feature_importances[f'fold_{fold+1}'] = model.feature_importances_; joblib.dump(model, os.path.join(CONFIG['MODEL_OUTPUT_DIR'], f'tabular_model_fold_{fold}.pkl'))
        
    feature_importances['mean'] = feature_importances.mean(axis=1); plot_feature_importance(feature_importances['mean'], os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'feature_importance.png'))
    overall_smape = smape(np.expm1(df_train[target]), np.expm1(oof_preds)); print(f"\nOverall OOF SMAPE with TUNED features: {overall_smape:.4f}")
    
    max_price = np.percentile(df_train['price'], 99.9); test_preds_orig_scale = np.expm1(test_preds); test_preds_orig_scale = np.clip(test_preds_orig_scale, 0, max_price)
    
    pd.DataFrame({'sample_id': df_train['sample_id'], 'oof_prediction_tabular': np.expm1(oof_preds)}).to_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'tabular_model_oof.csv'), index=False)
    pd.DataFrame({'sample_id': df_test['sample_id'], 'price': test_preds_orig_scale}).to_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'tabular_model_submission.csv'), index=False)
    print("Outputs saved successfully.")