import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import os
import joblib

# --- Configuration ---
CONFIG = {
    'PROCESSED_DATA_DIR': 'processed_data/',
    'MODEL_OUTPUT_DIR': 'models/',
    'SUBMISSION_OUTPUT_DIR': 'submissions/',
    'TRAIN_EMBEDDING_FILE': 'image_embeddings_train.npy',
    'TEST_EMBEDDING_FILE': 'image_embeddings_test.npy', # Assumes this has 5,000 rows
    'N_SPLITS': 5,
    'RANDOM_STATE': 42,
    'RIDGE_ALPHA': 1.0
}

# Ensure directories exist
os.makedirs(CONFIG['MODEL_OUTPUT_DIR'], exist_ok=True)
os.makedirs(CONFIG['SUBMISSION_OUTPUT_DIR'], exist_ok=True)

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

if __name__ == '__main__':
    # --- 1. Load Data and PRE-COMPUTED Embeddings ---
    print("Loading data and pre-computed image embeddings...")
    df_train = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features.csv'))
    df_test_full = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features_test.csv'))
    
    train_embedding_path = os.path.join(CONFIG['PROCESSED_DATA_DIR'], CONFIG['TRAIN_EMBEDDING_FILE'])
    test_embedding_path = os.path.join(CONFIG['PROCESSED_DATA_DIR'], CONFIG['TEST_EMBEDDING_FILE'])

    try:
        train_embeddings = np.load(train_embedding_path)
        test_embeddings = np.load(test_embedding_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find embedding files. {e}")
        exit()

    # ==============================================================================
    # === THE KEY CHANGE: Handle the mismatch between test set and embeddings ===
    # ==============================================================================
    is_hybrid_prediction = False
    if len(df_test_full) != len(test_embeddings):
        is_hybrid_prediction = True
        
        print("\n" + "="*50)
        print("!!! WARNING: Mismatch Detected !!!")
        print(f"Full test set has {len(df_test_full)} samples, but you only have {len(test_embeddings)} test embeddings.")
        print("Proceeding with a HYBRID prediction strategy.")
        print("The first {len(test_embeddings)} predictions will be from this CV model.")
        print("The rest will use the NLP model's predictions as a fallback.")
        print("="*50 + "\n")
        
        # Use the NLP submission as our base
        try:
            fallback_submission_df = pd.read_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'nlp_model_submission.csv'))
        except FileNotFoundError:
            print("FATAL ERROR: The fallback 'nlp_model_submission.csv' is missing. Please run the NLP model first.")
            exit()
            
        # We only need to predict on the part of the test set we have embeddings for
        df_test_partial = df_test_full.head(len(test_embeddings))
    else:
        # If everything matches, proceed as normal
        df_test_partial = df_test_full

    # ==============================================================================

    target = 'price_log1p'
    y_train = df_train[target]

    # --- 2. Train Ridge Model ---
    print("\nTraining Ridge model on image embeddings...")
    kf = KFold(n_splits=CONFIG['N_SPLITS'], shuffle=True, random_state=CONFIG['RANDOM_STATE'])
    oof_preds = np.zeros(df_train.shape[0])
    # The size of test_preds array now matches the number of embeddings we actually have
    test_preds = np.zeros(len(df_test_partial))

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_embeddings)):
        print(f"--- Fold {fold+1}/{CONFIG['N_SPLITS']} ---")
        X_train_fold, X_val_fold = train_embeddings[train_idx], train_embeddings[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = Ridge(alpha=CONFIG['RIDGE_ALPHA'], random_state=CONFIG['RANDOM_STATE'])
        model.fit(X_train_fold, y_train_fold)
        
        val_preds = model.predict(X_val_fold)
        oof_preds[val_idx] = val_preds
        # This now works because test_preds and the prediction have the same shape
        test_preds += model.predict(test_embeddings) / CONFIG['N_SPLITS']
        joblib.dump(model, os.path.join(CONFIG['MODEL_OUTPUT_DIR'], f'cv_ridge_model_fold_{fold}.pkl'))

    # --- 3. Final Evaluation and Output ---
    # The OOF score is still 100% correct and valid because you have all the train embeddings
    overall_smape = smape(np.expm1(y_train), np.expm1(oof_preds))
    print(f"\nOverall OOF SMAPE for CV Ridge Model: {overall_smape:.4f}")

    oof_df = pd.DataFrame({'sample_id': df_train['sample_id'], 'oof_prediction_cv': np.expm1(oof_preds)})
    oof_df.to_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'cv_model_oof.csv'), index=False)
    print("CV OOF predictions saved (This file is complete and correct).")
    
    # --- 4. Create and Save the HYBRID Submission File ---
    print("\nCreating final submission file...")
    test_preds_orig_scale = np.expm1(test_preds)
    test_preds_orig_scale[test_preds_orig_scale < 0] = 0

    if is_hybrid_prediction:
        print("Applying hybrid strategy for submission file...")
        # Start with the full NLP submission as the base
        final_submission_df = fallback_submission_df.copy()
        
        # Create a temporary dataframe with the real CV predictions for the first 5k samples
        partial_cv_preds_df = pd.DataFrame({
            'sample_id': df_test_partial['sample_id'],
            'price': test_preds_orig_scale
        })
        
        # Overwrite the prices for the first 5k samples with our new, real predictions
        # Set index to sample_id to align the data correctly
        final_submission_df = final_submission_df.set_index('sample_id')
        partial_cv_preds_df = partial_cv_preds_df.set_index('sample_id')
        final_submission_df.update(partial_cv_preds_df)
        
        # Reset the index to get the 'sample_id' column back
        final_submission_df.reset_index(inplace=True)
        
    else:
        # If we had all embeddings, create the submission file normally
        final_submission_df = pd.DataFrame({'sample_id': df_test_full['sample_id'], 'price': test_preds_orig_scale})

    final_submission_df.to_csv(os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'cv_model_submission.csv'), index=False)
    print("CV submission file saved successfully.")
    print("Top 5 predictions:")
    print(final_submission_df.head())