import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
CONFIG = {
    'PROCESSED_DATA_DIR': 'processed_data/',
    'MODEL_OUTPUT_DIR': 'models/',
    'SUBMISSION_OUTPUT_DIR': 'submissions/',
    'N_SPLITS': 5,
    'RANDOM_STATE': 42
}

# Ensure directories exist
os.makedirs(CONFIG['MODEL_OUTPUT_DIR'], exist_ok=True)
os.makedirs(CONFIG['SUBMISSION_OUTPUT_DIR'], exist_ok=True)


def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100


def plot_feature_importance(feature_importances, output_path):
    df_imp = pd.DataFrame({'feature': feature_importances.index, 'importance': feature_importances.values})
    df_imp = df_imp.sort_values('importance', ascending=False).head(40)
    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=df_imp)
    plt.title('Top 40 Feature Importances (Ensemble Model)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importance plot saved to {output_path}")


if __name__ == '__main__':
    # --- 1. Load All Necessary Data ---
    print("Loading base features and OOF predictions from all models...")
    
    # Load base features
    try:
        df_train_base = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features.csv'))
        df_test_base = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features_test.csv'))
    except FileNotFoundError as e:
        print(f"Error: Base feature files not found. Run feature_engineering.py. Details: {e}")
        exit()

    # Load OOF predictions for the training set (these will be our new features)
    try:
        oof_tabular = pd.read_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'tabular_model_oof.csv'))
        oof_nlp = pd.read_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'nlp_lgbm_model_oof.csv'))
        
        # --- Placeholder for CV Model ---
        # Create a dummy CV OOF file for now. Replace this with your actual file.

        # we have to add now here cv cvs file which is trained on images and prediction in this cv_model_oof.csv file

        if not os.path.exists(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'cv_model_oof.csv')):
            print("Warning: cv_model_oof.csv not found. Creating a placeholder by copying the NLP OOF.")
            oof_cv = oof_nlp.copy()
            oof_cv.columns = ['sample_id', 'oof_prediction_cv']
        else:
            oof_cv = pd.read_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'cv_model_oof.csv'))
        # --------------------------------


    except FileNotFoundError as e:
        print(f"Error: OOF prediction file not found. Ensure all base models have been trained. Details: {e}")
        exit()

    # --- 2. Create the Level 1 Training Dataset ---
    print("Creating the ensemble training dataset...")
    
    # Rename OOF columns for clarity
    oof_tabular.rename(columns={'oof_prediction_tabular': 'pred_tabular'}, inplace=True)
    oof_nlp.rename(columns={list(oof_nlp.columns)[1]: 'pred_nlp'}, inplace=True)
    oof_cv.rename(columns={list(oof_cv.columns)[1]: 'pred_cv'}, inplace=True)

    # Merge OOF predictions with the base training data
    df_train = df_train_base.merge(oof_tabular, on='sample_id', how='left')
    df_train = df_train.merge(oof_nlp, on='sample_id', how='left')
    df_train = df_train.merge(oof_cv, on='sample_id', how='left')

    # --- 3. Create the Level 1 Test Dataset ---
    print("Creating the ensemble test dataset...")
    
    # Load the submission files from each base model to get their test set predictions
    try:
        sub_tabular = pd.read_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'tabular_model_submission.csv'))
        sub_nlp = pd.read_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'nlp_model_submission.csv')) # Assuming this is from the tuned LGBM version

        # --- Placeholder for CV Model Submission ---
        if not os.path.exists(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'cv_model_submission.csv')):
            print("Warning: cv_model_submission.csv not found. Creating a placeholder by copying the NLP submission.")
            sub_cv = sub_nlp.copy()
        else:
            sub_cv = pd.read_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'cv_model_submission.csv'))
        # ------------------------------------------

    except FileNotFoundError as e:
        print(f"Error: Submission file not found. Ensure all base models have generated test predictions. Details: {e}")
        exit()

    # Rename submission columns to match the new features
    sub_tabular.rename(columns={'price': 'pred_tabular'}, inplace=True)
    sub_nlp.rename(columns={'price': 'pred_nlp'}, inplace=True)
    sub_cv.rename(columns={'price': 'pred_cv'}, inplace=True)

    # Merge test predictions with the base test data
    df_test = df_test_base.merge(sub_tabular, on='sample_id', how='left')
    df_test = df_test.merge(sub_nlp, on='sample_id', how='left')
    df_test = df_test.merge(sub_cv, on='sample_id', how='left')

    # --- 4. Define Features, Target, and Model Parameters ---
    print("Defining features and model parameters...")
    
    # The features are the original tabular features PLUS the new prediction features
    features = [col for col in df_train_base.columns if col not in [
        'sample_id', 'image_link', 'text_for_embedding', 'price', 'price_log1p', 'brand'
    ]] + ['pred_tabular', 'pred_nlp', 'pred_cv']


    # we have to add now here cv cvs file which is trained on images and prediction in this cv_model_oof.csv file
    
    target = 'price_log1p'
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

    # A slightly regularized set of parameters is often best for the ensemble model
    ensemble_params = {
        'objective': 'huber', 'metric': 'mae', 'random_state': CONFIG['RANDOM_STATE'],
        'n_estimators': 2000, 'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': -1,
        'learning_rate': 0.01, 'num_leaves': 31, 'max_depth': 7,
        'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 1,
        'lambda_l1': 0.5, 'lambda_l2': 0.5, 'min_child_samples': 30
    }

    # --- 5. Train the Final Ensemble Model ---
    print("\nTraining the final ensemble model...")
    kf = KFold(n_splits=CONFIG['N_SPLITS'], shuffle=True, random_state=CONFIG['RANDOM_STATE'])
    oof_preds = np.zeros(df_train.shape[0])
    test_preds = np.zeros(df_test.shape[0])
    feature_importances = pd.DataFrame(index=features)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        print(f"--- Ensemble Fold {fold+1}/{CONFIG['N_SPLITS']} ---")
        X_train, X_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
        y_train, y_val = X_train[target], X_val[target]
        
        model = lgb.LGBMRegressor(**ensemble_params)
        model.fit(X_train[features], y_train,
                  eval_set=[(X_val[features], y_val)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        
        val_preds = model.predict(X_val[features])
        oof_preds[val_idx] = val_preds
        test_preds += model.predict(df_test[features]) / CONFIG['N_SPLITS']
        
        feature_importances[f'fold_{fold+1}'] = model.feature_importances_
        joblib.dump(model, os.path.join(CONFIG['MODEL_OUTPUT_DIR'], f'ensemble_model_fold_{fold}.pkl'))

    # --- 6. Final Evaluation and Submission ---
    feature_importances['mean'] = feature_importances.mean(axis=1)
    plot_feature_importance(feature_importances['mean'], os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'ensemble_feature_importance.png'))

    overall_smape = smape(np.expm1(df_train[target]), np.expm1(oof_preds))
    print(f"\nFINAL ENSEMBLE OOF SMAPE: {overall_smape:.4f}")

    max_price = np.percentile(df_train['price'], 99.9)
    test_preds_orig_scale = np.expm1(test_preds)
    test_preds_orig_scale = np.clip(test_preds_orig_scale, 0, max_price)

    submission_df = pd.DataFrame({'sample_id': df_test['sample_id'], 'price': test_preds_orig_scale})
    submission_df.to_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'final_ensemble_submission.csv'), index=False)
    print("Final submission file saved successfully!")