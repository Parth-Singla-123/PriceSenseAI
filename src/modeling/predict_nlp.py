import pandas as pd
import numpy as np
import os
import joblib

# --- Configuration ---
CONFIG = {
    'PROCESSED_DATA_DIR': 'processed_data/',
    'MODEL_OUTPUT_DIR': 'models/',
    'SUBMISSION_OUTPUT_DIR': 'submissions/',
    'EMBEDDING_FILE': 'text_embeddings.npz',
    'N_SPLITS': 5, # Must match the number of models you trained
    'MODEL_PREFIX': 'nlp_lgbm_model_fold_' # The prefix for your saved model files
}

# Ensure the output directory exists
os.makedirs(CONFIG['SUBMISSION_OUTPUT_DIR'], exist_ok=True)

if __name__ == '__main__':
    # --- 1. Load the Necessary Data for Prediction ---
    print("Loading test data and pre-computed embeddings...")
    
    # Load the processed test set to get the sample_ids
    try:
        df_test = pd.read_csv(os.path.join(CONFIG['PROCESSED_DATA_DIR'], 'processed_features_test.csv'))
    except FileNotFoundError:
        print(f"Error: test_features.csv not found. Please run the feature engineering script first.")
        exit()

    # Load the pre-computed embeddings for the test set
    try:
        embeddings = np.load(os.path.join(CONFIG['PROCESSED_DATA_DIR'], CONFIG['EMBEDDING_FILE']))
        test_embeddings = embeddings['test']
    except FileNotFoundError:
        print(f"Error: Embedding file not found at {os.path.join(CONFIG['PROCESSED_DATA_DIR'], CONFIG['EMBEDDING_FILE'])}")
        print("Please run the 'generate_embeddings.py' or the combined training script first.")
        exit()

    print(f"Test embeddings loaded with shape: {test_embeddings.shape}")

    # --- 2. The Prediction Loop ---
    print("\nStarting prediction using the 5 trained NLP models...")
    
    # Initialize an array to store the predictions from all models
    test_preds = np.zeros(test_embeddings.shape[0])

    for fold in range(CONFIG['N_SPLITS']):
        # Construct the path to the model for the current fold
        model_filename = f"{CONFIG['MODEL_PREFIX']}{fold}.pkl"
        model_path = os.path.join(CONFIG['MODEL_OUTPUT_DIR'], model_filename)
        
        try:
            # Load the pre-trained model
            print(f"Loading model: {model_filename}...")
            model = joblib.load(model_path)
            
            # Predict on the test embeddings and add to the total
            fold_preds = model.predict(test_embeddings)
            test_preds += fold_preds
            
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            print("Please ensure you have run the 'train_nlp_tuned.py' script and that the models are saved.")
            exit()
    
    # --- 3. Finalize Predictions ---
    print("\nAveraging predictions from all models...")
    # Average the predictions by dividing by the number of folds
    test_preds /= CONFIG['N_SPLITS']
    
    print("Converting predictions from log scale back to original price scale...")
    # The models were trained on log(1+price), so we must apply the inverse function
    test_preds_orig_scale = np.expm1(test_preds)
    
    # A safety check to ensure no negative prices are predicted
    test_preds_orig_scale[test_preds_orig_scale < 0] = 0
    
    # --- 4. Create and Save the Submission File ---
    print("Creating submission file...")
    submission_df = pd.DataFrame({
        'sample_id': df_test['sample_id'],
        'price': test_preds_orig_scale
    })
    
    submission_filename = os.path.join(CONFIG['SUBMISSION_OUTPUT_DIR'], 'nlp_model_submission.csv')
    submission_df.to_csv(submission_filename, index=False)
    
    print(f"\nSubmission file successfully created and saved to: {submission_filename}")
    print("Top 5 predictions:")
    print(submission_df.head())